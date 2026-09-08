import json
import numpy as np

max_steps = 20

infile = "logs/alfworld_old/Qwen_Qwen3-4B-Instruct-2507/traj_eval_in_distribution_adamem-max_correct_only.json"
with open(infile, 'r') as reader:
    items = json.load(reader)

print("items[0]['steps'][0]['curr_prompt']:", items[0]["steps"][0]["curr_prompt"])
print("items[140]['steps'][0]['curr_prompt']:", items[140]["steps"][0]["curr_prompt"])
print("items[280]['steps'][0]['curr_prompt']:", items[280]["steps"][0]["curr_prompt"])

# find last step with valid prompt, all step before is failure, all step after is success
# verify first 30 tasks

runs = [items[0:140], items[140:280], items[280:]]
success_rates = []
for items in runs:
    rewards = []
    for item in items:
        last_idx = None
        for step in reversed(item["steps"]):
            if step["curr_prompt"] != "None":
                last_idx = step["step_idx"]
                break
        if last_idx is None:
            last_idx = len(item["steps"])
        # if last_idx < max_steps (zero-based), then success, else failure
        reward = 1 if last_idx < max_steps else 0
        rewards.append(reward)
    success_rate = sum(rewards) / len(rewards)
    success_rates.append(success_rate)

# print avg and std (ddof=1) of success rates
print("Success Rates:", success_rates)
print("Average Success Rate:", np.mean(success_rates))
print("Standard Deviation of Success Rates:", np.std(success_rates, ddof=1))
    