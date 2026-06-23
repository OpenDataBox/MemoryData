import uuid

from typing import Any

from pydantic import BaseModel, Field


class ParametricMemoryItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    memory: Any
    metadata: dict = {}
