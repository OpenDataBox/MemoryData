"""
Main entry point for the KV-cached memory agent system.
This demonstrates a simple conversation loop with memory storage and retrieval.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_validated_config
from src.conversation_manager.factory import create_chat_manager

# Try to load from config.yaml
try:
    # Ensure we're reading from project root
    project_root = Path(__file__).parent.parent
    config_path = project_root / 'config.yaml'
    APP_CONFIG = load_validated_config(str(config_path))
    MODEL_ID = APP_CONFIG.get_model_label()
    CHAT_MANAGER_KWARGS = APP_CONFIG.to_chat_manager_kwargs()
    STORAGE_MODE = APP_CONFIG.memory.storage_mode
    CLEAN_CACHE_ON_START = APP_CONFIG.memory.clean_cache_first
    ROUTER_SYSTEM_PROMPT = APP_CONFIG.memory.router_system_prompt
    MAX_NEW_TOKENS = 1024
except Exception as e:
    print(f"Warning: Could not load config.yaml: {e}")
    print("Using default configuration.\n")
    MODEL_ID = "Qwen/Qwen3-4B"
    CHAT_MANAGER_KWARGS = {
        "storage_mode": "kv_cache",
        "model_id": MODEL_ID,
        "chat_openai_config": {"api_key": "your-api-key-here"},
        "aggregator_openai_config": {"api_key": "your-api-key-here"},
        "router_openai_config": {"api_key": "your-api-key-here"},
        "model_context_window": 32768,
        "attn_implementation": "sdpa",
        "device_map": "auto",
        "quantization_config": None,
        "max_memory": None,
        "clean_cache_first": True,
        "router_system_prompt": None,
    }
    STORAGE_MODE = "kv_cache"
    CLEAN_CACHE_ON_START = True
    ROUTER_SYSTEM_PROMPT = None
    MAX_NEW_TOKENS = 1024


def main():
    
    print("=" * 60)
    print("KV-Cached Memory Agent System")
    print("=" * 60)
    print(f"\nStorage Mode: {STORAGE_MODE}")
    print(f"Model: {MODEL_ID}")
    print("\nInitializing chat manager...")
    
    # Initialize chat manager using factory
    try:
        chat_manager = create_chat_manager(**CHAT_MANAGER_KWARGS)
    except Exception as e:
        print(f"\nError initializing chat manager: {e}")
        print("\nPlease check your config.yaml")
        return
    
    print("Chat manager initialized successfully!")
    print("\nYou can now chat with the assistant.")
    print("The assistant will automatically store and retrieve memories.")
    print("Type 'quit' or 'exit' to end the conversation.\n")
    
    # Conversation loop
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nGoodbye!")
                break
            
            # Chat with auto memory management
            print("\nAssistant: ", end="", flush=True)
            response = chat_manager.chat(
                user_input=user_input,
                auto_save=False,  # Let agent decide when to save
                save_original_input=False,  # Save processed memory, not raw input
                max_new_tokens=MAX_NEW_TOKENS
            )
            
            if response:
                print(response)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Continuing...")


if __name__ == "__main__":
    main()
