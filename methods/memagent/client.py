import aiohttp


class OpenAICompatibleClient:
    def __init__(self, base_url: str, api_key: str, timeout: int = 86400):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    async def chat_completion(
        self,
        model: str,
        message: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "api-key": self.api_key,
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": message}],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if "qwen3" in str(model or "").lower():
            payload["enable_thinking"] = False
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            ) as response:
                response.raise_for_status()
                data = await response.json()
                message_payload = data["choices"][0]["message"]
                content = message_payload.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, str) and item.strip():
                            text_parts.append(item.strip())
                        elif isinstance(item, dict):
                            text_value = item.get("text") or item.get("content")
                            if isinstance(text_value, str) and text_value.strip():
                                text_parts.append(text_value.strip())
                    if text_parts:
                        return "\n".join(text_parts)
                reasoning = message_payload.get("reasoning_content") or message_payload.get("reasoning")
                if isinstance(reasoning, str) and reasoning.strip():
                    return reasoning.strip()
                return ""
