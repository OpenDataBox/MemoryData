import logging
import re

logger = logging.getLogger(__name__)

def limit_memory_segments(response: str, max_segments: int = 5) -> str:
    """
    Limit the number of memory segments in the response.
    
    Args:
        response (str): The raw XML response from the LLM.
        max_segments (int): The maximum number of segments to keep.
        
    Returns:
        str: The modified response with limited segments.
    """
    if not response or max_segments <= 0:
        return response
        
    # Check if it's a retrieval response
    if "<response_type>retrieval</response_type>" not in response:
        return response
        
    # Find the relevant_memories block
    # Use re.DOTALL to match across newlines
    # We look for the block containing memory segments
    memories_match = re.search(r'(<relevant_memories>)(.*?)(</relevant_memories>)', response, re.DOTALL)
    if not memories_match:
        return response
        
    start_tag, memories_content, end_tag = memories_match.groups()
    
    # Find all segments
    # Pattern to match <memory_segment>...</memory_segment>
    # We use a non-greedy match (.*?) to find individual segments
    # We accept any content inside the tag
    segment_pattern = r'<memory_segment>.*?</memory_segment>'
    segments = re.findall(segment_pattern, memories_content, re.DOTALL)
    
    if len(segments) <= max_segments:
        return response
        
    logger.info(f"Limiting memory segments from {len(segments)} to {max_segments}")
    
    # Keep top segments
    kept_segments = segments[:max_segments]
    
    # Reconstruct the memories block content
    # We join them with newlines and add some indentation
    new_memories_content = "\n    " + "\n    ".join(kept_segments) + "\n"
    
    # Reconstruct the full block
    new_block = f"{start_tag}{new_memories_content}{end_tag}"
    
    # Replace the old block with the new one in the original response
    # We use replace with the exact matched string
    new_response = response.replace(memories_match.group(0), new_block)
    
    return new_response
