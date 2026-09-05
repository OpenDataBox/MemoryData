"""Example comparing KV Cache and Text Storage modes."""
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
    BASE_CHAT_MANAGER_KWARGS = APP_CONFIG.to_chat_manager_kwargs()
except Exception as e:
    print(f"Error loading config: {e}")
    exit(1)


def demo_mode(storage_mode: str):
    """Demo a specific storage mode."""
    print("\n" + "=" * 60)
    print(f"{storage_mode.upper().replace('_', ' ')} MODE")
    print("=" * 60)
    
    manager_kwargs = BASE_CHAT_MANAGER_KWARGS.copy()
    manager_kwargs["storage_mode"] = storage_mode
    manager_kwargs["clean_cache_first"] = True
    manager = create_chat_manager(**manager_kwargs)
    
    # Add memory
    print("\nAdding memory: 'My favorite color is blue.'")
    response = manager.chat("My favorite color is blue.", auto_save=True)
    print(f"Response: {response}")
    
    # Query memory
    print("\nQuerying: 'What is my favorite color?'")
    response = manager.chat("What is my favorite color?", auto_save=False)
    print(f"Response: {response}")


def main():
    print("=" * 60)
    print("Storage Mode Comparison")
    print("=" * 60)
    print("\nThis example demonstrates both storage modes using config.yaml")
    
    # Demo KV Cache mode
    demo_mode("kv_cache")
    
    # Demo Text Storage mode
    demo_mode("text")
    
    print("\n" + "=" * 60)
    print("Comparison completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
