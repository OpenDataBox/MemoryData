import json
import os
from typing import Dict, List


def get_metadata_file() -> str:
    """Get metadata file path, supporting override via environment variable."""
    kv_data_dir = os.environ.get('KV_DATA_DIR', os.path.join(os.getcwd(), "kv_data"))
    os.makedirs(kv_data_dir, exist_ok=True)
    return os.path.join(kv_data_dir, "agents_metadata.json")


# For backward compatibility
METADATA_FILE = get_metadata_file()


def save_agents_metadata(agents_data: List[Dict]):
    """Save agents metadata to JSON file."""
    metadata_file = get_metadata_file()
    os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(agents_data, f, ensure_ascii=False, indent=2)


def load_agents_metadata() -> List[Dict]:
    """Load agents metadata from JSON file."""
    metadata_file = get_metadata_file()
    if not os.path.exists(metadata_file):
        return []
    with open(metadata_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def clear_metadata():
    """Clear metadata for current session only."""
    metadata_file = get_metadata_file()
    if not os.path.exists(metadata_file):
        return
    
    # Get session_id from environment
    session_id = os.environ.get('EVAL_SESSION_ID', 'default')
    
    # Load all metadata
    all_metadata = load_agents_metadata()
    
    # Keep only metadata from other sessions
    other_sessions_metadata = [m for m in all_metadata if m.get("session_id") != session_id]
    
    # Save back
    save_agents_metadata(other_sessions_metadata)
    print(f"Cleared metadata for session: {session_id}")
