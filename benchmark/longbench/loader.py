"""LongBench dataset loader for repository evaluation runs."""

from pathlib import Path

from benchmark.memoryagentbench.hf_datasets import load_from_disk


def load_longbench_eval_data(dataset_config):
    """Load LongBench rows and convert them into the shared evaluation schema."""
    dataset_path = _resolve_longbench_dataset_path(dataset_config)
    max_test_samples = dataset_config.get("max_test_samples")

    rows = _load_longbench_rows(dataset_path, split_name=str(dataset_config.get("longbench_split") or "train"))
    rows = _filter_longbench_rows(rows, dataset_config)

    if max_test_samples:
        rows = rows[: int(max_test_samples)]

    return {"data": [_convert_longbench_row(row, dataset_config) for row in rows]}


def _resolve_longbench_dataset_path(dataset_config):
    test_file = str(dataset_config.get("test_files") or "").strip()
    if test_file:
        return Path(test_file)

    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "datasets" / "longBench" / "datasets"


def _load_longbench_rows(dataset_path: Path, split_name: str):
    dataset_path = Path(dataset_path)

    if dataset_path.is_dir():
        loaded_dataset = load_from_disk(str(dataset_path))
        if hasattr(loaded_dataset, "keys"):
            if split_name not in loaded_dataset:
                raise ValueError(
                    f"LongBench split '{split_name}' not found at {dataset_path}. "
                    f"Available splits: {list(loaded_dataset.keys())}"
                )
            return list(loaded_dataset[split_name])
        return list(loaded_dataset)

    if dataset_path.is_file():
        raise ValueError(
            f"LongBench dataset path '{dataset_path}' points to a file, but the slimmed project "
            "only supports a HuggingFace save_to_disk directory for LongBench."
        )

    raise FileNotFoundError(
        f"LongBench dataset path not found: {dataset_path}. "
        "Expected a HuggingFace save_to_disk directory."
    )


def _filter_longbench_rows(rows, dataset_config):
    include_lengths = _normalize_filter_values(dataset_config.get("longbench_lengths"))
    include_difficulties = _normalize_filter_values(dataset_config.get("longbench_difficulties"))
    include_domains = _normalize_filter_values(dataset_config.get("longbench_domains"))
    include_sub_domains = _normalize_filter_values(dataset_config.get("longbench_sub_domains"))
    include_ids = _normalize_filter_values(dataset_config.get("longbench_ids"))

    filtered_rows = []
    for row in rows:
        if include_lengths and str(row.get("length", "")).strip() not in include_lengths:
            continue
        if include_difficulties and str(row.get("difficulty", "")).strip() not in include_difficulties:
            continue
        if include_domains and str(row.get("domain", "")).strip() not in include_domains:
            continue
        if include_sub_domains and str(row.get("sub_domain", "")).strip() not in include_sub_domains:
            continue
        if include_ids and str(row.get("_id", "")).strip() not in include_ids:
            continue
        filtered_rows.append(row)

    return filtered_rows


def _normalize_filter_values(raw_value):
    if raw_value in (None, ""):
        return set()

    if isinstance(raw_value, (list, tuple, set)):
        values = raw_value
    else:
        values = str(raw_value).split(",")

    return {
        str(value).strip()
        for value in values
        if str(value).strip()
    }


def _convert_longbench_row(row, dataset_config):
    sample_id = str(row.get("_id", "")).strip()
    question = str(row.get("question", "") or "")
    answer = str(row.get("answer", "") or "").strip().upper()
    context = str(row.get("context", "") or "")

    domain = str(row.get("domain", "") or "")
    sub_domain = str(row.get("sub_domain", "") or "")
    difficulty = str(row.get("difficulty", "") or "")
    length = str(row.get("length", "") or "")

    return {
        "source": str(dataset_config.get("sub_dataset") or "longbench"),
        "context": context,
        "context_length": len(context),
        "questions": [question],
        "answers": [answer],
        "question_ids": [sample_id],
        "qa_pair_ids": [sample_id],
        "question_types": [sub_domain],
        "domain": domain,
        "sub_domain": sub_domain,
        "difficulty": difficulty,
        "length": length,
        "choice_A": str(row.get("choice_A", "") or ""),
        "choice_B": str(row.get("choice_B", "") or ""),
        "choice_C": str(row.get("choice_C", "") or ""),
        "choice_D": str(row.get("choice_D", "") or ""),
        "eval_metadata": [
            {
                "dataset": "longbench",
                "_id": sample_id,
                "domain": domain,
                "sub_domain": sub_domain,
                "difficulty": difficulty,
                "length": length,
                "choice_A": str(row.get("choice_A", "") or ""),
                "choice_B": str(row.get("choice_B", "") or ""),
                "choice_C": str(row.get("choice_C", "") or ""),
                "choice_D": str(row.get("choice_D", "") or ""),
            }
        ],
    }
