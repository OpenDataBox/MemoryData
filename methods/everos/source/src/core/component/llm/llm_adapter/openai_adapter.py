from typing import Dict, Any, List, Union, AsyncGenerator
import os
import openai
from core.component.llm.llm_adapter.completion import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from core.component.llm.llm_adapter.llm_backend_adapter import LLMBackendAdapter
from core.constants.errors import ErrorMessage


class OpenAIAdapter(LLMBackendAdapter):
    """OpenAI API adapter (implemented based on the official openai package)"""

    def __init__(self, config: Dict[str, Any]):
        # Save configuration
        self.config = config
        self.api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
        self.base_url = config.get("base_url") or os.getenv("OPENAI_BASE_URL")
        self.timeout = config.get("timeout", 600)

        if not self.api_key:
            raise ValueError(ErrorMessage.INVALID_PARAMETER.value)

        # Instantiate openai async client
        self.client = openai.AsyncOpenAI(
            api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
        )

    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> Union[ChatCompletionResponse, AsyncGenerator[str, None]]:
        """
        Perform chat completion, supporting both streaming and non-streaming modes.
        """
        if not request.model:
            raise ValueError(ErrorMessage.INVALID_PARAMETER.value)

        params = request.to_dict()
        # The request `to_dict` method already filters for None values, but we can be explicit here for clarity
        # for what the openai client expects.
        client_params = {
            "model": params.get("model"),
            "messages": params.get("messages"),
            "temperature": params.get("temperature"),
            "max_tokens": params.get("max_tokens"),
            "top_p": params.get("top_p"),
            "frequency_penalty": params.get("frequency_penalty"),
            "presence_penalty": params.get("presence_penalty"),
            "stream": params.get("stream", False),
        }
        # Remove None values to avoid openai errors
        final_params = {k: v for k, v in client_params.items() if v is not None}

        try:
            if final_params.get("stream"):
                # Streaming response, return async generator
                async def stream_gen():
                    response_stream = await self.client.chat.completions.create(
                        **final_params
                    )
                    async for chunk in response_stream:
                        content = getattr(chunk.choices[0].delta, "content", None)
                        if content:
                            yield content

                return stream_gen()
            else:
                # Non-streaming response
                response = await self.client.chat.completions.create(**final_params)
                return ChatCompletionResponse.from_dict(response.model_dump())
        except Exception as e:
            raise RuntimeError(f"OpenAI chat completion request failed: {e}")

    def get_available_models(self) -> List[str]:
        """Get available model list (can be extended to call openai model list API)"""
        return self.config.get("models", [])
