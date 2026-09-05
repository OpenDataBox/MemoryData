"""Minimal OpenAI-compatible clients; no silent model or tokenizer fallbacks."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request

import numpy as np


class Tokenizer:
    def __init__(self, encoding="cl100k_base"):
        import tiktoken
        self.encoding = encoding
        self.codec = tiktoken.get_encoding(encoding)

    def count(self, text):
        return len(self.codec.encode(text, disallowed_special=()))

    def truncate(self, text, budget):
        if budget <= 0:
            return ""
        return self.codec.decode(self.codec.encode(text, disallowed_special=())[:budget])


def tokens(text):
    return re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE)


def json_object(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


class OpenAIEndpoint:
    def __init__(self, base_url, model, api_key_env="OPENAI_API_KEY", timeout=120, retries=2,
                 disable_thinking=False):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.key_env = api_key_env
        self.timeout = timeout
        self.retries = retries
        self.disable_thinking = disable_thinking
        self.last_attempts = 0

    @property
    def fingerprint(self):
        return self.model

    def post(self, route, body):
        key = os.environ.get(self.key_env, "")
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = "Bearer " + key
        request = urllib.request.Request(self.base_url + route, json.dumps(body).encode(), headers)
        self.last_attempts = 0
        for attempt in range(self.retries + 1):
            self.last_attempts = attempt + 1
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504} or attempt == self.retries:
                    raise RuntimeError(f"Model request failed: HTTP {exc.code} at {self.base_url}{route}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt == self.retries:
                    raise RuntimeError(f"Model endpoint unavailable: {self.base_url}{route}") from exc
            time.sleep(min(2 ** attempt, 4))
        raise RuntimeError("Request retries exhausted")

    def chat(self, messages, *, max_tokens=2048, json_mode=False, ledger=None, stage="generation"):
        body = {"model": self.model, "messages": messages, "temperature": 0.0, "max_tokens": max_tokens}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.disable_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        start = time.perf_counter()
        try:
            response = self.post("/chat/completions", body)
        except Exception:
            if ledger:
                ledger.cost({"stage": stage, "kind": "llm", "calls": self.last_attempts,
                             "missing_usage_calls": self.last_attempts, "seconds": time.perf_counter()-start})
            raise
        usage = response.get("usage", {})
        if ledger:
            reported = "prompt_tokens" in usage and "completion_tokens" in usage
            ledger.cost({"stage": stage, "kind": "llm", "calls": self.last_attempts, "input_tokens": usage.get("prompt_tokens", 0),
                         "output_tokens": usage.get("completion_tokens", 0), "seconds": time.perf_counter()-start,
                         "usage_reported": reported, "missing_usage_calls": self.last_attempts-int(reported)})
        choice = response["choices"][0]
        if choice.get("finish_reason") == "length" and json_mode:
            raise ValueError("Model JSON was truncated; increase generation_max_tokens")
        content = choice["message"].get("content")
        if not isinstance(content, str):
            raise ValueError("Model returned no textual content")
        return json_object(content) if json_mode else content


class RemoteEmbedder:
    def __init__(self, endpoint: OpenAIEndpoint, store, batch_size=32):
        self.endpoint = endpoint
        self.store = store
        self.batch_size = batch_size

    @property
    def fingerprint(self):
        return self.endpoint.fingerprint

    def encode(self, texts, stage="embedding"):
        texts = list(texts)
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        keys = [hashlib.sha256((self.fingerprint + "\0" + t).encode()).hexdigest() for t in texts]
        cache, missing = {}, {}
        hits = 0
        for key, text in zip(keys, texts):
            row = self.store.db.execute("SELECT payload FROM embeddings WHERE key=?", (key,)).fetchone()
            if row:
                cache[key] = json.loads(row[0])
                hits += 1
            else:
                missing[key] = text
        entries = list(missing.items())
        for begin in range(0, len(entries), self.batch_size):
            batch = entries[begin:begin+self.batch_size]
            start = time.perf_counter()
            try:
                response = self.endpoint.post("/embeddings", {"model": self.endpoint.model, "input": [t for _, t in batch]})
            except Exception:
                self.store.cost({"stage": stage, "kind": "embedding", "calls": self.endpoint.last_attempts,
                                 "missing_usage_calls": self.endpoint.last_attempts, "seconds": time.perf_counter()-start})
                raise
            data = sorted(response["data"], key=lambda x: x["index"])
            if [d["index"] for d in data] != list(range(len(batch))):
                raise ValueError("Embedding response length mismatch")
            self.store.cost({"stage": stage, "kind": "embedding", "calls": self.endpoint.last_attempts,
                             "missing_usage_calls": self.endpoint.last_attempts-int("total_tokens" in response.get("usage", {})),
                             "usage_reported": "total_tokens" in response.get("usage", {}),
                             "input_tokens": response.get("usage", {}).get("total_tokens", 0),
                             "seconds": time.perf_counter()-start})
            with self.store.db:
                for (key, _), item in zip(batch, data):
                    vector = np.asarray(item["embedding"], dtype=np.float32)
                    if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all():
                        raise ValueError("Invalid embedding vector")
                    vector /= max(float(np.linalg.norm(vector)), 1e-12)
                    cache[key] = vector.tolist()
                    self.store.db.execute("INSERT OR REPLACE INTO embeddings VALUES (?,?)", (key, json.dumps(cache[key])))
        if hits:
            self.store.cost({"stage": stage, "cache_hits": hits})
        return np.asarray([cache[k] for k in keys], dtype=np.float32)
