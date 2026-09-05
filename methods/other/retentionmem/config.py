"""Build one isolated runtime from an explicit, recorded configuration."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .backends import OpenAIEndpoint, RemoteEmbedder, Tokenizer
from .engine import RetentionMemory
from .features import FeatureExtractor, RuleRisk
from .generation import CandidateGenerator, TransitionVerifier
from .policy import RetentionPolicy
from .storage import MemoryStore


def load_config(path):
    import yaml
    with Path(path).open(encoding="utf-8") as stream:
        result = yaml.safe_load(stream)
    if not isinstance(result, dict):
        raise ValueError("Configuration must be a mapping")
    return result


def endpoint(config, kind="llm"):
    section = config[kind]
    return OpenAIEndpoint(section["base_url"], section["model"], section.get("api_key_env", "OPENAI_API_KEY"),
                          timeout=section.get("timeout", 120), disable_thinking=section.get("disable_thinking", False))


def build_runtime(config, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(directory/"memory.sqlite")
    try:
        return _assemble(config, store)
    except Exception:
        store.close()
        raise


def _assemble(config, store):
    manifest = json.loads(json.dumps(config))
    checkpoint = config.get("policy_checkpoint")
    if checkpoint:
        file = Path(checkpoint)
        manifest["checkpoint_sha256"] = hashlib.sha256(file.read_bytes()).hexdigest() if file.is_file() else "missing"
    previous = store.meta("runtime_manifest")
    if previous is not None and previous != manifest:
        raise ValueError("Saved memory belongs to a different configuration/checkpoint; use a new output directory")
    store.set_meta("runtime_manifest", manifest)
    tokenizer = Tokenizer(config.get("tokenizer_encoding", "cl100k_base"))
    embedder = RemoteEmbedder(endpoint(config, "embedding"), store)
    mode = config.get("mode", "learned_residual")
    features = FeatureExtractor(embedder, tokenizer)
    policy = None
    if mode.startswith("learned"):
        policy = RetentionPolicy.load(config.get("policy_checkpoint", ""), features.fingerprint)
    elif mode.startswith("heuristic"):
        policy = RuleRisk()
    llm = endpoint(config)
    generator = CandidateGenerator(llm, store, style=config.get("compressor", "structured"),
                                   max_probes=config.get("max_probes", 8), max_tokens=config.get("generation_max_tokens", 3072))
    verifier = TransitionVerifier(endpoint(config, "judge"), store) if mode == "criteria_verifier" else None
    engine = RetentionMemory(store, embedder, tokenizer, generator, policy, mode=mode, verifier=verifier,
                             region_size=config.get("region_size", 8), buffer_size=config.get("buffer_size", 8),
                             compression_ratio=config.get("compression_ratio", 4),
                             max_source_tokens=config.get("max_source_tokens", 24000), sparse_weight=config.get("sparse_weight", .5))
    return engine
