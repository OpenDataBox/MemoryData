from dataclasses import dataclass, field

@dataclass
class Config:
    model: str = "Qwen/Qwen3-8B"
    embed_model: str = "Qwen/Qwen3-Embedding-0.6B"   # switch to 4B if VRAM allows
    max_model_len: int = 16384
    gpu_memory_utilization: float = 0.88
    locomo_path: str = "data/locomo10.json"
    out_dir: str = "runs/4_2a"
    host: str = "raw"            # raw | mem0 | lightmem
    probes_per_session: int = 10
    retrieve_k: int = 8
    context_budget_tokens: int = 4096
    seed: int = 0
    family_weights: dict = field(default_factory=lambda: {
        "date": 0.25, "negation": 0.25, "entity": 0.2, "condition": 0.15, "number": 0.15})
