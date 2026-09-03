import os
import hnswlib, numpy as np, pickle, json
import torch
import re
import json
import argparse

from utils import embed_texts

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Build index for dataset')
parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
parser.add_argument('--base_model_name', type=str, required=True, help='Name of the base model')
parser.add_argument('--step_stride', type=int, default=1, help='Keep one entry every K steps (default: 1)')
parser.add_argument('--correct_only', action='store_true', help='Only index successful episodes')
parser.add_argument('--failure_only', action='store_true', help='Only index failed episodes')
args = parser.parse_args()
if args.step_stride < 1:
    raise ValueError(f"step_stride must be >= 1, got {args.step_stride}")
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
index_path = f"{retrieval_data_path}/train_hnsw.index"
meta_path  = f"{retrieval_data_path}/train_hnsw_meta.json"
values_path = f"{retrieval_data_path}/train_values.pkl"
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
with open(traj_file, 'r') as reader:
    items = json.load(reader)

pattern = r'\[.*?\]\n'
action_pattern = r"<action>(.*?)</action>"
observation_pattern = r"your current observation is:\s*(.*?)\s*Your admissible actions"
keys, values = [], [] 
for item in items:
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
            
            break

    reward = "success" if item["won"] else "failure"
    if correct_only and reward == "failure":
        continue
    if failure_only and reward == "success":
        continue

    valid_steps = []
    for step_idx, step in enumerate(item["steps"]):
        if step["curr_action"] == "None" or step["curr_prompt"] == "None":
            continue
        valid_steps.append((step_idx, step))

    for valid_idx, (step_idx, step) in enumerate(valid_steps):
        if valid_idx % args.step_stride != 0:
            continue
        keys.append(step["curr_prompt"])
        action_match = re.search(action_pattern, step["curr_action"], flags=re.DOTALL)
        if action_match:
            action_text = action_match.group(1).strip()
        else:
            action_text = ""

        start_idx = min(step_idx, len(observation_action_pairs))
        remaining_trajectory = '\n'.join(observation_action_pairs[start_idx:])

        experience_item = f"""
Past State:
{step['curr_prompt']}

Past Action:
{action_text}

Remaining Trajectory:
{remaining_trajectory}

Final Outcome:
{reward}
"""
        values.append(experience_item)

# -------------------------------
# 2) Embedding model (Qwen3-Embedding-4B) via server
# -------------------------------

print(f"Key 0: {keys[0]}")
print(f"Value 0: {values[0]}")
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


