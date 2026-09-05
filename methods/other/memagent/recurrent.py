import asyncio
from dataclasses import dataclass
from typing import Optional

from transformers import AutoTokenizer

from .client import OpenAICompatibleClient
from .prompt import FINAL_ANSWER_TEMPLATE, NO_MEMORY, UPDATE_MEMORY_TEMPLATE


def clip_long_string(text: str, max_length: int = 2000) -> str:
    if len(text) <= max_length:
        return text
    marker = "\n\n...(truncated)\n\n"
    target_len = max_length - len(marker)
    return text[: target_len // 2] + marker + text[-target_len // 2 :]


@dataclass
class RecurrentConfig:
    model: str
    base_url: str
    api_key: str = "123-abc"
    tokenizer_model: Optional[str] = None
    recurrent_chunk_size: int = 5000
    recurrent_max_new: int = 1024
    recurrent_max_context_len: int = 120000
    temperature: float = 0.7
    top_p: float = 0.95


class MemAgentRecurrentRunner:
    def __init__(self, config: RecurrentConfig):
        self.config = config
        self.client = OpenAICompatibleClient(
            base_url=config.base_url,
            api_key=config.api_key,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_model or config.model,
            trust_remote_code=True,
        )

    async def arun(self, context: str, prompt: str, verbose: bool = False) -> str:
        input_ids = self.tokenizer.encode(context.strip())
        max_len = self.config.recurrent_max_context_len
        if len(input_ids) > max_len:
            input_ids = input_ids[: max_len // 2] + input_ids[-max_len // 2 :]

        memory = NO_MEMORY
        for start in range(0, len(input_ids), self.config.recurrent_chunk_size):
            chunk_ids = input_ids[start : start + self.config.recurrent_chunk_size]
            chunk_text = self.tokenizer.decode(chunk_ids)
            message = UPDATE_MEMORY_TEMPLATE.format(
                prompt=prompt.strip(),
                memory=memory,
                chunk=chunk_text,
            )
            if verbose:
                print("user:")
                print(clip_long_string(message))
            memory = await self.client.chat_completion(
                model=self.config.model,
                message=message,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.recurrent_max_new,
            )
            if verbose:
                print("assistant:")
                print(clip_long_string(memory))

        final_message = FINAL_ANSWER_TEMPLATE.format(
            prompt=prompt.strip(),
            memory=memory,
        )
        if verbose:
            print("user:")
            print(clip_long_string(final_message))

        answer = await self.client.chat_completion(
            model=self.config.model,
            message=final_message,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.recurrent_max_new,
        )
        if verbose:
            print("assistant:")
            print(clip_long_string(answer))
        return answer

    def run(self, context: str, prompt: str, verbose: bool = False) -> str:
        return asyncio.run(self.arun(context=context, prompt=prompt, verbose=verbose))
