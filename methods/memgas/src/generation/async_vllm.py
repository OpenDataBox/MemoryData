
import aiohttp
import os
import asyncio

import csv
from tqdm import tqdm
import json

URL = os.getenv("URL", "http://localhost:9000/v1")
API_KEY = os.getenv("API_KEY", "123-abc")

async def async_query_llm(msg, model, temperature=1, top_p=0.95):

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=86400)) as session:
        
        try:
            async with session.post(
                url=URL + "/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}", "api-key": API_KEY},
                json=dict(
                    model=model, messages=[{"role": "user", "content": msg}], temperature=temperature, top_p=top_p, max_tokens=2560
                ),
            ) as resp:
                status = resp.status
                if status != 200:
                    print(f"{status=}, {model=}")
                    return ""
                data = await resp.json()
                return data["choices"][0]["message"]["content"]
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            import traceback
            
            traceback.print_exc()
            return ""

from tqdm.asyncio import tqdm as tqdm_asyncio

async def process_multiple_items(async_prompts, model):
    semaphore = asyncio.Semaphore(4)
    async def limited_async_query(msg):
        async with semaphore:
            return await async_query_llm(msg, model)
    tasks = [limited_async_query(msg) for msg in async_prompts]

    results = await tqdm_asyncio.gather(*tasks, desc="Processing vllm inference ...", total=len(async_prompts))

    return results


if __name__ == "__main__":
    import argparse
    prompts = ['who are you?','what you like?']
    parser = argparse.ArgumentParser(description="quick start")
    parser.add_argument("--model", type=str, required=True, help="model name used in your deployment/model service endpoint")
    args = parser.parse_args()
    async_responses = asyncio.run(process_multiple_items(prompts, args.model))

    print(async_responses)
    # how to use: 
    # CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen2.5-7B-Instruct --tensor_parallel_size 4 --port 9000
    # python async_vllm.py --model Qwen/Qwen2.5-7B-Instruct
