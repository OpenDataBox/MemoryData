"""Warm the official HF cache while the inference environment installs."""
from huggingface_hub import snapshot_download

for model in ['Qwen/Qwen3-8B', 'Qwen/Qwen3-Embedding-0.6B']:
    print('PREFETCH ' + model, flush=True)
    print(snapshot_download(model, ignore_patterns=['*.bin', '*.gguf', '*.onnx', 'onnx/*']), flush=True)
print('PREFETCH_READY', flush=True)
