import hnswlib, numpy as np, pickle, json
import torch
import re
import json
import asyncio
import random
import argparse

from openai import AsyncOpenAI, APITimeoutError
import os
import httpx

from tqdm.asyncio import tqdm

from utils import embed_texts

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Build index for dataset')
parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
parser.add_argument('--base_model_name', type=str, required=True, help='Name of the base model')
parser.add_argument('--correct_only', action='store_true', help='Only index successful episodes')
parser.add_argument('--failure_only', action='store_true', help='Only index failed episodes')
args = parser.parse_args()
if args.correct_only and args.failure_only:
    raise ValueError("--correct_only and --failure_only are mutually exclusive")

correct_only = args.correct_only
failure_only = args.failure_only

# -------------------------------
# 1) Your key/value data
# -------------------------------
dataset_name = args.dataset_name
base_model_name = args.base_model_name
# Sanitize base_model_name for file paths
base_model_safe = base_model_name.replace('/', '_')
traj_file = f"logs/{dataset_name}_old/{base_model_safe}/traj_train.json"
retrieval_data_path = f"retrieval_data/{dataset_name}/{base_model_safe}"
os.makedirs(retrieval_data_path, exist_ok=True)
index_path = f"{retrieval_data_path}/train_reasoningbank_hnsw.index"
meta_path  = f"{retrieval_data_path}/train_reasoningbank_hnsw_meta.json"
values_path = f"{retrieval_data_path}/train_reasoningbank_values.pkl"
if correct_only:
    index_path = index_path.replace(".index", "_correct_only.index")
    meta_path = meta_path.replace(".json", "_correct_only.json")
    values_path = values_path.replace(".pkl", "_correct_only.pkl")
elif failure_only:
    index_path = index_path.replace(".index", "_failure_only.index")
    meta_path = meta_path.replace(".json", "_failure_only.json")
    values_path = values_path.replace(".pkl", "_failure_only.pkl")
print(f"Building index from {traj_file}...")
print(f"Index will be saved to {index_path}...")

servers = os.environ.get("VLLM_SERVERS", "127.0.0.1").split(",")
clients = {}
async def call_llm(prompt, model_name="langfeng01/GiGPO-Qwen2.5-7B-Instruct-WebShop"):
    ip_addr = random.choice(servers)
    if ip_addr not in clients:
        clients[ip_addr] = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),  # vLLM doesn't check this
            base_url=f"http://{ip_addr}:8001/v1",
            timeout=httpx.Timeout(600.0),  # 10 minutes timeout
        )
    client = clients[ip_addr]
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            return resp.choices[0].message.content.strip()
        except (APITimeoutError, httpx.ReadTimeout) as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # exponential backoff
            else:
                raise e

# -------------------------------
# 1) Your key/value data
# -------------------------------
async def main():
    with open(traj_file, 'r') as reader:
        items = json.load(reader)

    pattern = r'\[.*?\]\n'
    action_pattern = r"<action>(.*?)</action>"
    observation_pattern = r"your current observation is:\s*(.*?)\s*Your admissible actions"
    keys, values = [], [] 
    prompts = []
    for item in items:
        task_instr = item["steps"][0]["curr_prompt"]
        reward = "success" if item["won"] else "failure"
        if correct_only and reward == "failure":
            continue
        if failure_only and reward == "success":
            continue

        for step in reversed(item["steps"]):
            if step["curr_prompt"] != "None":
                full_traj = step["curr_prompt"]
                observation_action_pairs = [p for p in re.findall(pattern, full_traj, flags=re.DOTALL) if p.startswith("[Observation")]
                last_action = step["curr_action"]
                action_match = re.search(action_pattern, last_action, flags=re.DOTALL)
                if action_match:
                    action_text = action_match.group(1).strip()
                else:
                    action_text = ""
                observation_match = re.search(observation_pattern, full_traj, flags=re.IGNORECASE | re.DOTALL)
                if observation_match:
                    observation_text = observation_match.group(1).strip()
                else:
                    observation_text = ""

                observation_action_pairs.append(f"[Last Observation: '{observation_text}', Last Action: '{action_text}]'")

                experience = f"Task:\n{task_instr}\n\nAgent Trajectory:\n{observation_action_pairs}\n\nFinal Outcome: {reward}"
                
                break
        
        prompt = f"""
You will be given a task description and a corresponding trajectory that represents **how an agent attempted to accomplish the task** — the trajectory may be either **successful** or **failed**.

Your task is to extract and summarize **useful, generalizable insights** in the format of **memory items**, depending on whether the agent’s trajectory was successful or not.

---

## **Guidelines**

* If the trajectory is **successful**, summarize insights that explain **why and how the agent succeeded**.
* If the trajectory is **failed**, summarize insights that explain **why the agent failed** and **what strategies could prevent such failures in the future**.
* The goal of the summarized memory items is to help improve performance on future similar tasks by providing **generalizable lessons** (not specific to the query or domain).

---

## **Important Notes**

* First, **reflect on the trajectory’s outcome** (success or failure) and reason about **why it happened**.
* You can extract **at most 3** memory items per trajectory.
* Avoid repeating or overlapping items.
* Do **not** mention task-specific information. Focus on **general strategies, reasoning patterns, or process insights**.

---

## **Output Format**

Your output must strictly follow the Markdown format below:

```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary of the memory item>
## Content <1–3 sentences describing the insights learned from the trajectory (for success: what worked well; for failure: what to improve or avoid)>
```

"""
        prompts.append(prompt + experience)

    memory_items = [None] * len(prompts)
    semaphore = asyncio.Semaphore(300)  # Limit to 300 concurrent requests

    async def call_with_limit(p):
        async with semaphore:
            return await call_llm(p)

    with tqdm(total=len(prompts), desc="Processing LLM calls") as pbar:
        tasks = [asyncio.create_task(call_with_limit(p)) for p in prompts]
        for i, task in enumerate(asyncio.as_completed(tasks)):
            memory_items[i] = await task
            pbar.update(1)
    for memory_item in memory_items:
        # Split the response into individual memory items
        items = re.split(r'(?=^# Memory Item \d+)', memory_item.strip(), flags=re.MULTILINE)
        for item in items:
            item = item.strip()
            if item:
                # Remove the "# Memory Item X" header line
                lines = item.split('\n')
                if lines and re.match(r'# Memory Item \d+', lines[0]):
                    item = '\n'.join(lines[1:]).strip()
                if item:
                    keys.append(item)
                    values.append(item)

    # -------------------------------
    # 2) Embedding model (Qwen3-Embedding-4B) via server
    # -------------------------------
    print("sample key", keys[10])
    print("sample value", values[10])
    print(f"num of entries: {len(keys)}")

    emb = embed_texts(keys, model="Qwen/Qwen3-Embedding-4B", normalize=True)
    emb = torch.tensor(emb, dtype=torch.float32)
    emb = emb.cpu().contiguous().numpy().astype("float32")
    d = emb.shape[1]

    # -------------------------------
    # 3) Build HNSW index (cosine)
    # -------------------------------

    # M = graph degree, ef_construction = build quality/latency trade-off
    M, ef_construction, ef_search = 32, 200, 100

    index = hnswlib.Index(space="cosine", dim=d)
    index.init_index(max_elements=len(emb), ef_construction=ef_construction, M=M)
    index.add_items(emb, ids=np.arange(len(emb), dtype=np.int64))
    index.set_ef(ef_search)  # query-time recall/speed dial

    # -------------------------------
    # 4) Persist
    # -------------------------------
    index.save_index(index_path)
    with open(meta_path, "w") as f:
        json.dump({"dim": d, "M": M, "ef_construction": ef_construction, "ef_search": ef_search}, f)

    with open(values_path, "wb") as f:
        pickle.dump(values, f)

if __name__ == "__main__":
    asyncio.run(main())


