import json
import os
import sys
from transformers import AutoTokenizer
from tqdm import tqdm
from glob import glob

if len(sys.argv) != 4:
    print("Usage: python calculate_trajectory_stats.py <dataset_name> <mem_type> <model_name>")
    sys.exit(1)

dataset_name = sys.argv[1]
mem_type = sys.argv[2]
model_name = sys.argv[3]

base_dir = f"logs/{dataset_name}/{model_name.replace('/', '_')}"
# List of trajectory files to process
if mem_type == "no_memory":
    if dataset_name == "alfworld_old":
        file_list = [
            os.path.join(base_dir, "traj_eval_in_distribution.json"),
            # os.path.join(base_dir, "traj_eval_out_of_distribution.json")
        ]
    elif dataset_name == "webshop":
        file_list = [os.path.join(base_dir, "traj_eval.json")]
else:
    if dataset_name == "alfworld_old":
        file_list = [
            os.path.join(base_dir, f"traj_eval_in_distribution_{mem_type}.json"),
            # os.path.join(base_dir, f"traj_eval_out_of_distribution_{mem_type}.json")
        ]
    elif dataset_name == "webshop":
        print(f"{base_dir}/traj_eval_{mem_type}.json")
        print(glob(f"{base_dir}/traj_eval_{mem_type}.json"))
        file_list = [sorted(glob(f"{base_dir}/traj_eval_{mem_type}.json"))[-1]]


# Initialize tokenizer for Qwen model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Thinking-2507")

# Determine input and output fields based on mem_type
if mem_type == "adamem-high":
    input_fields = ["initial_prompt", "strategy_prompt", "action_prompt"]
    output_fields = ["initial_response", "strategy_response", "final_response"]
elif mem_type in ["adamem-max", "adamem-max-without-trajectory-memory"]:
    input_fields = ["strategy_prompt", "action_prompt"]
    output_fields = ["strategy_response", "final_response"]
elif mem_type == "adamem-low":
    input_fields = ["refresh_prompt", "strategy_prompt", "action_prompt"]
    output_fields = ["refresh_response", "strategy_response", "final_response"]
elif mem_type == "adamem-max-without-strategy-memory":
    input_fields = ["action_prompt"]
    output_fields = ["final_response"]
else:
    # no-memory, synapse, reasoningbank
    input_fields = ["curr_prompt"]
    output_fields = ["curr_action"]

print("input_fields", input_fields)
print("output_fields", output_fields)
# Lists to collect statistics
num_actions_list = []
total_input_tokens = 0
total_output_tokens = 0
total_actions = 0

total_memory_tokens = 0
total_action_tokens = 0

# Process each file
for filename in file_list:
    filepath = filename # os.path.join(base_dir, filename)
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        continue
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    for traj in tqdm(data, desc=f"Processing {filename}"):
        steps = traj['steps']
        # Filter out steps where curr_action is None
        valid_steps = [step for step in steps if step.get('curr_action') != 'None']
        num_actions = len(valid_steps)
        num_actions_list.append(num_actions)
        
        input_tokens_traj = 0
        output_tokens_traj = 0
        memory_tokens_traj = 0
        action_tokens_traj = 0
        
        for step_idx, step in enumerate(valid_steps):
            if mem_type == "adamem-low":
                if step["refresh_prompt"] is None:
                    # first step, always refresh
                    assert step_idx == 0
                    refresh = True
                elif step["action_prompt"] is None:
                    # skip strategy generation, no refresh
                    refresh = False
                else:
                    # both strategy and action generation
                    refresh = True

            # Count input tokens
            for field in input_fields:
                if field in step and step[field] and step[field] != 'None':
                    token_cnt = len(tokenizer.encode(step[field]))
                    input_tokens_traj += token_cnt
                    if mem_type == "adamem-low":
                        if field == "strategy_prompt":
                            memory_tokens_traj += token_cnt
                        elif field == "action_prompt":
                            action_tokens_traj += token_cnt
                        elif field == "refresh_prompt":
                            if refresh:
                                memory_tokens_traj += token_cnt
                            else:
                                action_tokens_traj += token_cnt
                        else:
                            assert False, f"Unexpected field {field} for mem_type {mem_type}"
                    else:
                        # For other mem_types, all input tokens are action tokens
                        action_tokens_traj += token_cnt

            # Count output tokens
            for idx, field in enumerate(output_fields):
                # need to ensure that corresponding input field is also present
                if field in step and step[field] and step[input_fields[idx]] and step[field] != 'None':
                    token_cnt = len(tokenizer.encode(step[field]))
                    output_tokens_traj += token_cnt
                    if mem_type == "adamem-low":
                        if field == "strategy_response":
                            memory_tokens_traj += token_cnt
                        elif field == "final_response":
                            action_tokens_traj += token_cnt
                        elif field == "refresh_response":
                            if refresh:
                                memory_tokens_traj += token_cnt
                            else:
                                action_tokens_traj += token_cnt
                    else:
                        # For other mem_types, all output tokens are action tokens
                        action_tokens_traj += token_cnt

        
        # Accumulate totals
        total_input_tokens += input_tokens_traj
        total_output_tokens += output_tokens_traj
        total_actions += num_actions
        total_memory_tokens += memory_tokens_traj
        total_action_tokens += action_tokens_traj

# Calculate averages
if num_actions_list:
    num_traj = len(num_actions_list)
    avg_num_actions = sum(num_actions_list) / len(num_actions_list)
    if total_actions > 0:
        avg_input_tokens_per_action = total_input_tokens / total_actions
        avg_output_tokens_per_action = total_output_tokens / total_actions
    else:
        avg_input_tokens_per_action = 0
        avg_output_tokens_per_action = 0

    assert total_input_tokens + total_output_tokens == total_memory_tokens + total_action_tokens, "Token count mismatch!"
    
    print(f"Total trajectories processed: {len(num_actions_list)}")
    print(f"Average number of actions per trajectory: {avg_num_actions:.1f}")
    print(f"Average input tokens per action: {avg_input_tokens_per_action:.0f}")
    print(f"Average output tokens per action: {avg_output_tokens_per_action:.0f}")
    print(f"Average total tokens per action: {(avg_input_tokens_per_action + avg_output_tokens_per_action):.0f}")
    print("---- Memory vs Action Tokens ----")
    print(f"Average memory tokens per trajectory: {(total_memory_tokens / num_traj):.1f}")
    print(f"Average action tokens per trajectory: {(total_action_tokens / num_traj):.1f}")
    print(f"Average total tokens per trajectory: {((total_memory_tokens + total_action_tokens) / num_traj):.1f}")
else:
    print("No trajectories found.")
