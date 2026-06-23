import re


def remove_thinking_tags(text: str) -> str:
    """
    Remove thinking tags from the generated text.

    Args:
        text: The generated text.

    Returns:
        str: The cleaned text.
    """
    return re.sub(r"^<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
