from benchmark.memoryagentbench.hf_datasets import get_dataset_class
from benchmark.memoryagentbench.hf_datasets import load_dataset
from benchmark.memoryagentbench.hf_datasets import load_from_disk
from benchmark.locomo.loader import load_locomo_eval_data
from benchmark.longbench.loader import load_longbench_eval_data
from benchmark.membench.loader import load_membench_eval_data
from pathlib import Path

import logging

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ============================================================================
# HUGGING FACE DATASET LOADING
# ============================================================================

def _get_local_memoryagentbench_path():
    """Return the local MemoryAgentBench save_to_disk path if available."""
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / "datasets" / "MemoryAgentBench" / "eval_dataset_collection"
    return candidate if candidate.exists() and any(candidate.iterdir()) else None


def load_data_huggingface(dataset_name, sub_dataset_source, max_test_samples=None, seed=42):
    """
    Args:
        dataset_name: The dataset name (Accurate_Retrieval, Test_Time_Learning,
                     Conflict_Resolution)
        sub_dataset_source: The sub_dataset name used to filter by source field
        max_test_samples: Maximum number of test samples to load
        seed: Random seed for sampling

    Returns:
        Dictionary with processed data
    """
    print(f"Loading {sub_dataset_source} from Hugging Face dataset: ai-hyz/MemoryAgentBench")

    # Configuration for Hugging Face dataset
    huggingface_dataset_name = "ai-hyz/MemoryAgentBench"

    # Supported dataset splits (identity mapping)
    supported_splits = {
        "Accurate_Retrieval", "Test_Time_Learning",
        "Conflict_Resolution"
    }

    # Validate dataset name
    if dataset_name not in supported_splits:
        raise ValueError(f"Unknown dataset {dataset_name}. Available splits: {sorted(supported_splits)}")

    split_name = dataset_name

    # Prefer a local save_to_disk copy to avoid re-downloading during evaluation.
    local_dataset_path = _get_local_memoryagentbench_path()
    if local_dataset_path is not None:
        try:
            dataset_dict = load_from_disk(str(local_dataset_path))
            raw_data = dataset_dict[split_name]
            print(f"Loaded {len(raw_data)} samples from local disk split '{split_name}' at {local_dataset_path}")

            original_length = len(raw_data)
            dataset = raw_data.filter(lambda sample: sample.get("metadata", {}).get("source", "") == sub_dataset_source)
            print(f"Filtered to {len(dataset)} samples matching source '{sub_dataset_source}' "
                  f"(from {original_length} total)")

            if max_test_samples is not None and len(dataset) > max_test_samples:
                dataset = dataset.select(range(max_test_samples))
                print(f"Subsampled to {max_test_samples} samples")
        except Exception as e:
            print(f"Warning: failed to load local MemoryAgentBench copy: {e}")
            print("Falling back to Hugging Face dataset loader.")
            dataset = _load_and_filter_dataset(
                huggingface_dataset_name,
                split_name,
                sub_dataset_source,
                max_test_samples,
                seed,
            )
    else:
        dataset = _load_and_filter_dataset(
            huggingface_dataset_name,
            split_name,
            sub_dataset_source,
            max_test_samples,
            seed,
        )

    processed_dataset = _process_qa_list_fields(dataset)

    return {"data": processed_dataset}



def _load_and_filter_dataset(dataset_name, split_name, source_filter, max_samples, seed):
    """Load dataset from HuggingFace and apply filtering and sampling."""
    try:
        # Load the specific split from HuggingFace
        raw_data = load_dataset(dataset_name, split=split_name, revision="main")
        print(f"Loaded {len(raw_data)} samples from {split_name}")

        # Filter by source to match the sub_dataset exactly (source is now in metadata)
        original_length = len(raw_data)
        filtered_data = raw_data.filter(lambda sample: sample.get("metadata", {}).get("source", "") == source_filter)
        print(f"Filtered to {len(filtered_data)} samples matching source '{source_filter}' "
              f"(from {original_length} total)")

        # Apply max_test_samples limit if specified
        if max_samples is not None and len(filtered_data) > max_samples:
            filtered_data = filtered_data.select(range(max_samples))
            print(f"Subsampled to {max_samples} samples")

        return filtered_data

    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise RuntimeError(
            f"Failed to load Hugging Face split '{split_name}' from '{dataset_name}'. "
            "This is often caused by blocked network access, an incomplete local cache, "
            "or a missing local dataset fallback."
        ) from e


def _process_qa_list_fields(dataset):
    """
    Process the dataset to ensure Q&A pairs and related fields are properly formatted as lists.

    Args:
        dataset: HuggingFace dataset object

    Returns:
        HuggingFace dataset with processed list fields
    """
    # Convert back to HuggingFace dataset format
    return get_dataset_class().from_list([_process_single_sample_qa_lists(sample) for sample in dataset])


def _process_single_sample_qa_lists(sample):
    """
    Process a single sample to ensure all Q&A related fields are properly formatted as lists.

    Args:
        sample: Single data sample dictionary

    Returns:
        Processed sample with list-formatted fields
    """
    # Process main Q&A fields
    metadata = sample.get("metadata", {})

    # Define metadata fields to process
    metadata_fields = ["question_dates", "question_types", "question_ids", "previous_events", "qa_pair_ids", "demo"]

    # Create processed sample with standardized list fields
    processed_sample = dict(sample)
    processed_sample.update({
        "questions": _ensure_field_is_list(sample["questions"]),
        "answers": _ensure_field_is_list(sample["answers"]),
        "source": metadata.get("source", ""),
        "eval_metadata": _build_eval_metadata_list(sample),
        **{field: _ensure_field_is_list(metadata.get(field, [])) for field in metadata_fields}
    })

    return processed_sample


def _build_eval_metadata_list(sample):
    """Create per-question metadata records for downstream evaluation/debugging."""
    metadata = sample.get("metadata", {})
    questions = _ensure_field_is_list(sample.get("questions", []))
    answers = _ensure_field_is_list(sample.get("answers", []))
    question_ids = _ensure_field_is_list(metadata.get("question_ids", []))
    qa_pair_ids = _ensure_field_is_list(metadata.get("qa_pair_ids", []))
    question_types = _ensure_field_is_list(metadata.get("question_types", []))
    question_dates = _ensure_field_is_list(metadata.get("question_dates", []))
    previous_events = _ensure_field_is_list(metadata.get("previous_events", []))
    source = metadata.get("source", "")

    num_items = max(
        len(questions),
        len(answers),
        len(question_ids),
        len(qa_pair_ids),
        len(question_types),
        len(question_dates),
        len(previous_events),
        1,
    )

    eval_metadata = []
    for index in range(num_items):
        question_id = question_ids[index] if index < len(question_ids) else None
        qa_pair_id = qa_pair_ids[index] if index < len(qa_pair_ids) else question_id
        eval_metadata.append(
            {
                "dataset": "memoryagentbench_qa",
                "source": source,
                "question_id": question_id,
                "qa_pair_id": qa_pair_id,
                "question_type": question_types[index] if index < len(question_types) else None,
                "question_date": question_dates[index] if index < len(question_dates) else None,
                "previous_event": previous_events[index] if index < len(previous_events) else None,
            }
        )

    return eval_metadata


def _ensure_field_is_list(field_value):
    """
    Ensure a field value is properly formatted as a list.

    Args:
        field_value: Value that should be converted to list format

    Returns:
        List representation of the field value
    """
    if isinstance(field_value, list):
        return field_value
    elif field_value:
        # Single value (string or other) - wrap in list
        return [field_value]
    else:
        # Empty or None value
        return []
# ============================================================================
# MAIN DATA LOADING INTERFACE
# ============================================================================

def load_eval_data(dataset_config):
    """
    Main interface for loading dataset based on configuration.

    Args:
        dataset_config: Dictionary containing dataset configuration parameters

    Returns:
        Loaded and processed dataset
    """
    # Extract configuration parameters
    config_params = (
        dataset_config['dataset'], dataset_config['sub_dataset'],
        dataset_config["max_test_samples"], dataset_config["seed"]
    )
    main_dataset_name, sub_dataset_name, max_test_samples, random_seed = config_params

    print(f"Dataset: {sub_dataset_name}")

    # Load data based on dataset type
    supported_hf_datasets = {
        'Accurate_Retrieval', 'Test_Time_Learning',
        'Conflict_Resolution'
    }

    if str(main_dataset_name).lower() == "locomo":
        return load_locomo_eval_data(dataset_config)
    if str(main_dataset_name).lower() == "longbench":
        return load_longbench_eval_data(dataset_config)
    if str(main_dataset_name).lower() == "membench":
        return load_membench_eval_data(dataset_config)

    # Check if it's a HuggingFace dataset
    if main_dataset_name in supported_hf_datasets:
        return load_data_huggingface(main_dataset_name, sub_dataset_name, max_test_samples, random_seed)

    raise ValueError(
        f"Dataset '{sub_dataset_name}' not found. "
        f"Supported MemoryAgentBench splits: {sorted(supported_hf_datasets)}. "
        "For offline MemoryAgentBench runs, provide datasets/MemoryAgentBench/eval_dataset_collection/ "
        "or make the Hugging Face dataset accessible."
    )


# ============================================================================
# CHAT FORMATTING UTILITIES
# ============================================================================

def format_chat(message, include_system=True, system_message="You are a helpful assistant."):
    """
    Format a message into chat format for language model consumption.

    Args:
        message: The user message content
        include_system: Whether to include system message in the chat
        system_message: The system message content

    Returns:
        List of message dictionaries in chat format
    """
    chat_messages = [{"role": "user", "content": message}]

    if include_system:
        chat_messages.insert(0, {"role": "system", "content": system_message})

    return chat_messages
