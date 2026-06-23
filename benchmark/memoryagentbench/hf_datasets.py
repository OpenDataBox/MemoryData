import importlib.machinery
import importlib.util
import sys
from pathlib import Path


_HF_DATASETS_MODULE = None


def _load_external_datasets_module():
    repo_root = Path(__file__).resolve().parents[2]
    search_paths = []
    for path_entry in sys.path:
        if not path_entry:
            continue
        try:
            if Path(path_entry).resolve() == repo_root:
                continue
        except OSError:
            pass
        search_paths.append(path_entry)

    spec = importlib.machinery.PathFinder.find_spec("datasets", search_paths)
    if spec is None or spec.loader is None:
        raise ImportError("HuggingFace datasets package is not available")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _get_hf_datasets_module():
    global _HF_DATASETS_MODULE
    if _HF_DATASETS_MODULE is None:
        _HF_DATASETS_MODULE = _load_external_datasets_module()
    return _HF_DATASETS_MODULE


def load_dataset(*args, **kwargs):
    return _get_hf_datasets_module().load_dataset(*args, **kwargs)


def load_from_disk(*args, **kwargs):
    return _get_hf_datasets_module().load_from_disk(*args, **kwargs)


def get_dataset_class():
    return _get_hf_datasets_module().Dataset
