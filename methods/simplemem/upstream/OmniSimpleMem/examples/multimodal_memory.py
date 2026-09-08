"""
OmniMem Multimodal Memory Example

Demonstrates storing and retrieving multimodal content (text + images).

Prerequisites:
    pip install omnimem[visual]
    export OPENAI_API_KEY=your_key_here
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omni_memory import OmniMemoryOrchestrator, OmniMemoryConfig


def main():
    # Configure for multimodal usage
    config = OmniMemoryConfig()
    config.embedding.model_name = "all-MiniLM-L6-v2"
    config.embedding.embedding_dim = 384
    config.embedding.visual_embedding_model = "UCSC-VLAA/openvision-vit-large-patch14-224"
    config.embedding.visual_embedding_dim = 768

    orchestrator = OmniMemoryOrchestrator(
        config=config,
        data_dir="./multimodal_data",
    )

    # Store text with image references
    orchestrator.add_text(
        "User showed a photo of their golden retriever Max playing in the park. "
        "image_caption: A golden retriever running through green grass with a tennis ball.",
        tags=["session_id:D1", "image_id:D1:IMG_001", "timestamp:2024-06-15"],
    )

    orchestrator.add_text(
        "User discussed dog training techniques. Max has learned to sit, stay, and fetch. "
        "They use positive reinforcement with treats.",
        tags=["session_id:D1", "timestamp:2024-06-15"],
    )

    orchestrator.add_text(
        "User shared another photo of Max at the beach. "
        "image_caption: A golden retriever swimming in ocean waves.",
        tags=["session_id:D2", "image_id:D2:IMG_001", "timestamp:2024-07-20"],
    )

    # Query about the dog
    print("Q: What tricks has Max learned?")
    result = orchestrator.query("What tricks has Max learned?", top_k=5)
    for item in result.items[:2]:
        print(f"  → {item.get('summary', '')[:120]}")

    print("\nQ: What does the user's dog look like?")
    result = orchestrator.query("What does the user's dog look like?", top_k=5)
    for item in result.items[:2]:
        print(f"  → {item.get('summary', '')[:120]}")

    orchestrator.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
