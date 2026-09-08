"""
OmniMem Quickstart Example

Demonstrates basic text memory operations: store conversations and query them.

Prerequisites:
    pip install omnimem
    export OPENAI_API_KEY=your_key_here
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omni_memory import OmniMemoryOrchestrator, OmniMemoryConfig


def main():
    # 1. Create configuration
    config = OmniMemoryConfig()
    config.embedding.model_name = "all-MiniLM-L6-v2"  # Local embedding (no API needed)
    config.embedding.embedding_dim = 384

    # 2. Initialize orchestrator
    orchestrator = OmniMemoryOrchestrator(
        config=config,
        data_dir="./quickstart_data",
    )

    # 3. Store conversation turns
    conversations = [
        {"text": "User mentioned they love hiking in the Rocky Mountains every summer.",
         "tags": ["session_id:D1", "timestamp:2024-06-15"]},
        {"text": "User discussed their new camera, a Sony A7IV, for landscape photography.",
         "tags": ["session_id:D1", "timestamp:2024-06-15"]},
        {"text": "User planned a trip to Yellowstone National Park next month.",
         "tags": ["session_id:D2", "timestamp:2024-07-01"]},
        {"text": "User bought a new telephoto lens (200-600mm) for wildlife photography.",
         "tags": ["session_id:D2", "timestamp:2024-07-01"]},
        {"text": "User shared photos from their Yellowstone trip — saw grizzly bears and bison.",
         "tags": ["session_id:D3", "timestamp:2024-08-10"]},
    ]

    print("Storing conversations...")
    for conv in conversations:
        orchestrator.add_text(conv["text"], tags=conv["tags"])
    print(f"Stored {len(conversations)} conversation turns.\n")

    # 4. Query the memory
    queries = [
        "What camera does the user have?",
        "Where did the user go hiking?",
        "What animals did the user see?",
        "What lens did the user buy?",
    ]

    for query in queries:
        print(f"Q: {query}")
        result = orchestrator.query(query, top_k=3)
        for item in result.items[:2]:
            summary = item.get("summary", "")[:100]
            print(f"  → {summary}")
        print()

    # 5. Cleanup
    orchestrator.close()
    print("Done!")


if __name__ == "__main__":
    main()
