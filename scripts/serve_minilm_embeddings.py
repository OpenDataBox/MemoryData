"""Serve the CPU SentenceTransformer encoder used by the HiGMem run."""

import argparse
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def encode_request(model, payload):
    if payload.get("model") not in {MODEL, "all-MiniLM-L6-v2"}:
        raise ValueError("The requested embedding model is not loaded")
    texts = payload.get("input")
    if isinstance(texts, str):
        texts = [texts]
    if not isinstance(texts, list) or not texts or not all(isinstance(t, str) for t in texts):
        raise ValueError("input must be a string or nonempty list of strings")
    if payload.get("dimensions", 384) != 384:
        raise ValueError("all-MiniLM-L6-v2 produces exactly 384 dimensions")
    encoding = payload.get("encoding_format", "float")
    if encoding not in {"float", "base64"}:
        raise ValueError("encoding_format must be float or base64")
    raw = model.tokenizer(texts, truncation=False, padding=False)["input_ids"]
    encoded = model.tokenize(texts)
    processed_tokens = int(encoded["attention_mask"].sum().item())
    original_tokens = sum(len(ids) for ids in raw)
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    records = []
    for index, vector in enumerate(vectors):
        value = (base64.b64encode(vector.astype("<f4").tobytes()).decode("ascii")
                 if encoding == "base64" else vector.tolist())
        records.append({"object": "embedding", "index": index, "embedding": value})
    return {
        "object": "list", "model": MODEL, "data": records,
        "usage": {
            "prompt_tokens": processed_tokens, "total_tokens": processed_tokens,
            "prompt_tokens_details": {
                "tokens_before_truncation": original_tokens,
                "encoded_tokens": processed_tokens,
                "truncated_inputs": sum(len(ids) > model.max_seq_length for ids in raw),
                "max_seq_length": model.max_seq_length,
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--model-path", default=MODEL)
    args = parser.parse_args()
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model_path, device="cpu")
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, data):
            body = json.dumps(data, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/v1/models":
                self.reply(200, {"object": "list", "data": [{"id": MODEL, "object": "model"}]})
            elif self.path == "/health":
                self.reply(200, {"ready": True, "model": MODEL, "dimensions": 384,
                                 "max_seq_length": model.max_seq_length, "device": "cpu"})
            else:
                self.reply(404, {"error": "Unknown endpoint"})

        def do_POST(self):
            if self.path != "/v1/embeddings":
                self.reply(404, {"error": "Unknown endpoint"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16 * 1024 * 1024:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError("Expected a JSON object")
                with lock:
                    result = encode_request(model, payload)
                self.reply(200, result)
            except (ValueError, TypeError) as exc:
                self.reply(400, {"error": {"message": str(exc)}})

    print(json.dumps({"ready": True, "host": args.host, "port": args.port,
                      "model": MODEL, "max_seq_length": model.max_seq_length}), flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
