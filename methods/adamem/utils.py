import json
import pickle
import threading
import time

import hnswlib
import numpy as np
import torch
import re
import os
import ray

import requests

EMB_VLLM_SERVER = os.environ["EMB_VLLM_SERVER"]
BASE_URL = f"http://{EMB_VLLM_SERVER}:8002/v1"
API_KEY  = "EMPTY"

def embed_texts(
    texts,
    model="Qwen/Qwen3-Embedding-4B",
    normalize=True,
    timeout=600,
    batch_size=256,
    max_retries=1,
):
    """Embed a string or list of strings via OpenAI-compatible embedding server."""
    if isinstance(texts, str):
        texts = [texts]

    if not isinstance(texts, (list, tuple)):
        raise TypeError(f"texts must be str or list/tuple of str, got {type(texts)}")
    if len(texts) == 0:
        raise ValueError("texts must not be empty")

    url = f"{BASE_URL}/embeddings"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    all_embs = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start:start + batch_size])
        payload = {"model": model, "input": batch}

        last_err = None
        for attempt in range(max_retries):
            try:
                r = requests.post(url, json=payload, headers=headers, timeout=timeout)
                r.raise_for_status()
                data = r.json()["data"]
                # Keep deterministic order even if server returns out-of-order entries.
                data = sorted(data, key=lambda x: x.get("index", 0))
                all_embs.extend([row["embedding"] for row in data])
                last_err = None
                break
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
                last_err = e
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
            except requests.exceptions.RequestException:
                # Non-retryable HTTP errors should fail fast.
                raise

        if last_err is not None:
            raise last_err

    X = np.array(all_embs, dtype=np.float32)
    if normalize:
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return X

mem_type = os.environ.get("MEM_TYPE", None)
is_mix_mode = os.environ.get("CORRECT_ONLY", "false").lower() == "mix"


def _get_file_paths(memory_type=None, split=None):
    """Get file paths based on memory type.

    split: None = use CORRECT_ONLY env var, "correct_only" = force success index,
           "failure_only" = force failure index.
    """
    dataset_name = os.environ["ENV_NAME"] # alfworld or webshop
    base_model_name = os.environ["MEMORY_MODEL_NAME"] # the model used for generating trajectory memory
    base_model_safe = base_model_name.replace('/', '_')
    selected_mem_type = memory_type or mem_type

    if selected_mem_type is None:
        return None, None, None

    ADAMEM_TYPES = {"adamem-high", "adamem-max", "adamem-max-without-trajectory-memory",
                    "adamem-max-without-strategy-memory", "adamem-low"}
    if selected_mem_type in ADAMEM_TYPES:
        index_path = f"retrieval_data/{dataset_name}/{base_model_safe}/train_hnsw.index"
        meta_path  = f"retrieval_data/{dataset_name}/{base_model_safe}/train_hnsw_meta.json"
        values_path = f"retrieval_data/{dataset_name}/{base_model_safe}/train_values.pkl"
    elif selected_mem_type == "synapse":
        index_path = f"retrieval_data/{dataset_name}/{base_model_safe}/train_traj_level_hnsw.index"
        meta_path  = f"retrieval_data/{dataset_name}/{base_model_safe}/train_traj_level_hnsw_meta.json"
        values_path = f"retrieval_data/{dataset_name}/{base_model_safe}/train_traj_level_values.pkl"
    elif selected_mem_type == "reasoningbank":
        index_path = f"retrieval_data/{dataset_name}/{base_model_safe}/train_reasoningbank_hnsw.index"
        meta_path  = f"retrieval_data/{dataset_name}/{base_model_safe}/train_reasoningbank_hnsw_meta.json"
        values_path = f"retrieval_data/{dataset_name}/{base_model_safe}/train_reasoningbank_values.pkl"
    else:
        return None, None, None

    # Determine which split suffix to apply.
    if split == "correct_only":
        effective_correct = True
        effective_failure = False
    elif split == "failure_only":
        effective_correct = False
        effective_failure = True
    else:
        # Use env var (backward compat).
        effective_correct = os.environ.get("CORRECT_ONLY", "false").lower() == "true"
        effective_failure = False

    if effective_failure:
        index_path = index_path.replace(".index", "_failure_only.index")
        meta_path = meta_path.replace(".json", "_failure_only.json")
        values_path = values_path.replace(".pkl", "_failure_only.pkl")
    elif effective_correct:
        index_path = index_path.replace(".index", "_correct_only.index")
        meta_path = meta_path.replace(".json", "_correct_only.json")
        values_path = values_path.replace(".pkl", "_correct_only.pkl")
    return index_path, meta_path, values_path

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'


def _get_task_instruction(memory_type=None):
    selected_mem_type = memory_type or mem_type
    if selected_mem_type == "reasoningbank":
        return "Given a task description and initial state of an agent, retrieve relevant strategies that can help solve the task."
    return "Given a task state of an agent, retrieve relevant states that are similar to the current one and may help solve the current task."

# Ray actor for centralized retrieval (memory-efficient)
@ray.remote
class RetrievalService:
    """Centralized retrieval service - loads index and values ONCE in a single actor"""

    def __init__(self, memory_type=None, split=None):
        self.memory_type = memory_type
        self.split = split
        index_path, meta_path, values_path = _get_file_paths(memory_type, split=split)
        
        if index_path is None:
            self.index = None
            self.values = None
            self.meta = None
            return
        
        print(f"[RetrievalService] Loading index from {index_path}...")
        with open(meta_path, "r") as f:
            self.meta = json.load(f)
        
        d_reload = self.meta["dim"]
        self.index = hnswlib.Index(space="cosine", dim=d_reload)
        self.index.load_index(index_path)
        self.index.set_ef(self.meta.get("ef_search", 100))
        print(f"[RetrievalService] Index loaded successfully")
        
        print(f"[RetrievalService] Loading values from {values_path}...")
        with open(values_path, "rb") as f:
            self.values = pickle.load(f)
        print(f"[RetrievalService] Values loaded successfully ({len(self.values)} items)")
    
    def query(self, embedding, topk=1):
        """Query the index with a pre-computed embedding"""
        if self.index is None or self.values is None:
            return []
        
        labels, dists = self.index.knn_query(embedding, k=topk)
        I = labels[0]
        D = 1.0 - dists[0]
        
        return [(self.values[idx], float(score)) for idx, score in zip(I, D)]

# Global retrieval service actors keyed by memory type
_retrieval_services = {}
_retrieval_lock = threading.Lock()

def _get_retrieval_service(memory_type=None, split=None):
    """Get or create a retrieval service actor for the requested memory type and split."""
    selected_mem_type = memory_type or mem_type

    if selected_mem_type is None:
        return None

    service_key = f"{selected_mem_type}:{split}" if split else selected_mem_type

    with _retrieval_lock:
        if service_key not in _retrieval_services:
            if not ray.is_initialized():
                ray.init()
            print(
                f"[INFO] Initializing centralized RetrievalService actor for {service_key}..."
            )
            _retrieval_services[service_key] = RetrievalService.remote(selected_mem_type, split=split)

    return _retrieval_services[service_key]

def get_top_k_memories(query_state, topk=1, memory_type=None, split=None):
    """Get top-k memories using a centralized retrieval service."""
    selected_mem_type = memory_type or mem_type

    if selected_mem_type is None:
        return []

    service = _get_retrieval_service(selected_mem_type, split=split)
    if service is None:
        return []

    task = _get_task_instruction(selected_mem_type)

    q = embed_texts([get_detailed_instruct(task, query_state)])

    results = ray.get(service.query.remote(q, topk=topk))

    return results


def get_top_k_memories_mix(query_state, topk=1, memory_type=None):
    """Retrieve topk from both success-only and failure-only indexes (for mix mode).

    Returns (success_results, failure_results), each a list of (value, score) tuples.
    """
    success_results = get_top_k_memories(query_state, topk=topk, memory_type=memory_type, split="correct_only")
    failure_results = get_top_k_memories(query_state, topk=topk, memory_type=memory_type, split="failure_only")
    return success_results, failure_results


def get_memories_by_types(query_state, memory_types, topk=1, split=None):
    """Get top-k memories for multiple memory types with a shared query state."""
    if not memory_types:
        return {}

    deduped_memory_types = list(dict.fromkeys(memory_types))
    return {
        memory_type: get_top_k_memories(
            query_state,
            topk=topk,
            memory_type=memory_type,
            split=split,
        )
        for memory_type in deduped_memory_types
    }

if __name__ == "__main__":
    print(get_top_k_memories("ALFWorld", topk=1)) 
