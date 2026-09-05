import os
import uuid

import torch


def get_kv_data_dir() -> str:
    """Get KV data directory, supporting override via environment variable."""
    data_dir = os.environ.get('KV_DATA_DIR', os.path.join(os.getcwd(), "kv_data"))
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


# For backward compatibility
KV_DATA_DIR = get_kv_data_dir()


class KVBlock:
    def __init__(self,block_id:uuid.UUID,create_timestamp:str,block_size:int=32000):
        self.block_id=block_id
        self.create_timestamp=create_timestamp
        self.store_target=os.path.join(get_kv_data_dir(), f"kv_cache_{self.block_id}_{self.create_timestamp}.pt")
        # block size measures the tokes that's stored instead of number of informations/chats
        # So the context window size should be at larger than the block size
        self.block_size=block_size
        self.block_used=0   # calc the used tokens
        self.chunk_num=0    # log the number of chunks stored in this block
        # initialize the kv cache block file only if it doesn't exist
        if not os.path.exists(self.store_target):
            torch.save({}, self.store_target)


    def save_cache(self,cache_state:dict,total_new_token:int):
        """
        Store the new kv cache to the block file.
        If the block is empty, just save the kv cache.
        If the block is not empty, append the new kv cache to the existing kv cache.
        Args:
            kv_cache (dict): A dict to be stored that contains the kv cache, global_offset and other information(the create step will be handled elsewhere)
            total_new_token (int): The total number of new tokens while creating the new kv cache consumed, used to analyze whether the block is full.
        
        Returns:
            bool: True if the block is full after saving and a new one needs to be created, False otherwise.
        """
        assert isinstance(cache_state, dict), "cache_state must be a dict"
        for key in cache_state.keys():
            assert key in ['global_offset','saved_chunks','chunk_number','model_id','merged_cache','original_texts'],"Unknown key found in the cache_state"
        torch.save(cache_state, self.store_target)
        if total_new_token > 0:
            self.block_used += total_new_token
        self.chunk_num=cache_state.get('chunk_number',0)
        if self.is_full():
            return True
        else:
            return False

    def load_cache(self):
        """Load cache state from disk."""
        if not os.path.exists(self.store_target):
            return {}
        try:
            cache_state = torch.load(self.store_target, weights_only=False)
            return cache_state if cache_state else {}
        except Exception as e:
            print(f"Error loading cache from {self.store_target}: {e}")
            return {}
    
    def is_full(self):
        """Check if block is full."""
        return self.block_used >= self.block_size



def clear_cache():
    """Clear kv cache files for current session only."""
    kv_data_dir = get_kv_data_dir()
    if not os.path.exists(kv_data_dir):
        return
    
    # Get session_id from environment
    session_id = os.environ.get('EVAL_SESSION_ID', 'default')
    
    # Load metadata to find which blocks belong to current session
    from src.memory.kv_block_manager.metadata import load_agents_metadata
    metadata = load_agents_metadata()
    
    # Get block_ids for current session
    current_session_blocks = set()
    for m in metadata:
        if m.get("session_id") == session_id:
            current_session_blocks.add(m.get("block_id"))
    
    # Delete only files belonging to current session
    for file in os.listdir(kv_data_dir):
        if file.endswith(".pt"):
            # Extract block_id from filename: kv_cache_{block_id}_{timestamp}.pt
            parts = file.replace("kv_cache_", "").replace(".pt", "").split("_")
            if parts:
                block_id = parts[0]
                if block_id in current_session_blocks:
                    print(f"Deleting {file} (session: {session_id})")
                    os.remove(os.path.join(kv_data_dir, file))
        

