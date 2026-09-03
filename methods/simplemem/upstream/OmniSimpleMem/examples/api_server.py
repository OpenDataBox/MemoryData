"""
OmniMem REST API Server Example

Launches a FastAPI server exposing OmniMem as a REST API.

Prerequisites:
    pip install omnimem[server]
    export OPENAI_API_KEY=your_key_here

Usage:
    python api_server.py
    # Then visit http://localhost:8000/docs for API documentation

API Endpoints:
    POST /memory/text     - Store text memory
    POST /memory/query    - Query memories
    GET  /memory/stats    - Get memory statistics
    GET  /health          - Health check
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    import uvicorn
    from omni_memory.app import app

    print("Starting OmniMem API server...")
    print("API docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
