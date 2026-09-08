"""Example demonstrating text storage mode."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_validated_config
from src.conversation_manager.factory import create_chat_manager

# Load config
try:
    project_root = Path(__file__).parent.parent
    config_path = project_root / 'config.yaml'
    APP_CONFIG = load_validated_config(str(config_path))
    CHAT_MANAGER_KWARGS = APP_CONFIG.to_chat_manager_kwargs()
except Exception as e:
    print(f"Error loading config: {e}")
    exit(1)


def main():
    print("=" * 60)
    print("Text Storage Mode Example")
    print("=" * 60)
    
    # Create text storage manager
    manager_kwargs = CHAT_MANAGER_KWARGS.copy()
    manager_kwargs["storage_mode"] = "text"
    manager_kwargs["clean_cache_first"] = True
    manager = create_chat_manager(**manager_kwargs)
    
    print("\n1. Adding memories...")
    memories = [
        "My name is Alice and I work at Google.",
        "I love programming in Python.",
        "My favorite hobby is hiking."
    ]
    
    for memory in memories:
        response = manager.chat(memory, auto_save=True)
        print(f"   Added: {memory}")
    
    print("\n2. Querying memories...")
    queries = [
        "What is my name?",
        "Where do I work?",
        "What do I like to do?"
    ]
    
    for query in queries:
        print(f"\n   Q: {query}")
        response = manager.chat(query, auto_save=False)
        print(f"   A: {response}")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
