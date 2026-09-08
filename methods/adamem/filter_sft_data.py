import json
import random
import argparse
import importlib
from glob import glob


SUBSET_CNT = 1e9 # 10000


def _avg_len_by_qwen_tokens(items, tokenizer):
    if not items:
        return 0.0
    return sum(len(tokenizer.encode(x["output"])) for x in items) / len(items)

def main():
    parser = argparse.ArgumentParser(description="Filter SFT data from trajectory file")
    parser.add_argument("--traj_file", required=True, help="Path pattern to the trajectory JSON file")
    parser.add_argument(
        "--mode", required=True,
        choices=["outcome-mft", "action_only", "strategy_and_action", "step-mft"],
        help=(
            "outcome-mft: keep strategies from all successful trajectories. "
            "step-mft: keep only strategies that changed the agent action (process-level filter). "
            "action_only / strategy_and_action: include action generation targets."
        ),
    )
    parser.add_argument("--output_file", default="sft_data.json", help="Output JSON file for SFT data")
    parser.add_argument("--rejected_output_file", default=None, help="Optional JSON file to save rejected strategies in step-mft mode")
    parser.add_argument("--tokenizer_name", default="Qwen/Qwen3-4B-Thinking-2507", help="Qwen tokenizer name for length counting")
    args = parser.parse_args()

    transformers_module = importlib.import_module("transformers")
    tokenizer = transformers_module.AutoTokenizer.from_pretrained(args.tokenizer_name)

    # Load trajectory data
    data = []
    for infile in glob(args.traj_file):
        print(f"Loading data from {infile}")
        with open(infile, 'r') as f:
            data.extend(json.load(f))

    sft_data = []
    accepted_step_mft = []
    rejected_success_no_change = []
    rejected_failure = []

    # Filter correct trajectories
    correct_trajectories = [traj for traj in data if traj.get("won") == True]

    for traj in correct_trajectories:
        steps = traj.get("steps", [])
        for step in steps:
            if args.mode in ["outcome-mft", "strategy_and_action"]:
                strategy_prompt = step.get("strategy_prompt")
                strategy_response = step.get("strategy_response")
                if strategy_prompt and strategy_response:
                    sft_data.append({
                        "instruction": strategy_prompt,
                        "input": "",
                        "output": strategy_response
                    })

            if args.mode == "step-mft":
                final_action = step.get("final_action", "")
                initial_action = step.get("initial_action", "")
                strategy_prompt = step.get("strategy_prompt")
                strategy_response = step.get("strategy_response")
                if strategy_prompt and strategy_response:
                    item = {
                        "instruction": strategy_prompt,
                        "input": "",
                        "output": strategy_response
                    }
                    if final_action != initial_action and not (final_action.startswith("search[") and initial_action.startswith("search[")):
                        sft_data.append({
                            "instruction": strategy_prompt,
                            "input": "",
                            "output": strategy_response
                        })
                        accepted_step_mft.append(item)
                    else:
                        rejected_success_no_change.append(item)

            if args.mode in ["action_only", "strategy_and_action"]:
                action_prompt = step.get("action_prompt")
                final_response = step.get("final_response")
                if action_prompt and final_response:
                    sft_data.append({
                        "instruction": action_prompt,
                        "input": "",
                        "output": final_response
                    })

    if args.mode == "step-mft":
        failed_trajectories = [traj for traj in data if traj.get("won") != True]
        for traj in failed_trajectories:
            steps = traj.get("steps", [])
            for step in steps:
                strategy_prompt = step.get("strategy_prompt")
                strategy_response = step.get("strategy_response")
                if strategy_prompt and strategy_response:
                    rejected_failure.append({
                        "instruction": strategy_prompt,
                        "input": "",
                        "output": strategy_response
                    })

    sft_data = random.sample(sft_data, min(SUBSET_CNT, len(sft_data)))
    # Save to output file
    with open(args.output_file, 'w') as f:
        json.dump(sft_data, f, indent=2)

    print(f"Total SFT items curated: {len(sft_data)}")
    print(f"Saved to {args.output_file}")

    if args.mode == "step-mft":
        rejected_all = rejected_success_no_change + rejected_failure
        print("\n[step-mft stats]")
        print(f"Accepted (success + action changed): {len(accepted_step_mft)}")
        print(f"Rejected (success but no action change): {len(rejected_success_no_change)}")
        print(f"Rejected (all failure trajectories): {len(rejected_failure)}")
        print(f"Rejected (all): {len(rejected_all)}")

        accepted_avg_tokens = _avg_len_by_qwen_tokens(accepted_step_mft, tokenizer)
        rejected_avg_tokens = _avg_len_by_qwen_tokens(rejected_all, tokenizer)
        print(f"Accepted avg output length (Qwen tokens): {accepted_avg_tokens:.2f}")
        print(f"Rejected avg output length (Qwen tokens): {rejected_avg_tokens:.2f}")

        if accepted_avg_tokens > rejected_avg_tokens:
            print("Accepted strategies are longer on average (by Qwen tokens).")
        else:
            print("Accepted strategies are NOT longer on average (by Qwen tokens).")

        if args.rejected_output_file:
            with open(args.rejected_output_file, 'w') as f:
                json.dump({
                    "rejected_success_no_change": rejected_success_no_change,
                    "rejected_failure": rejected_failure,
                    "rejected_all": rejected_all
                }, f, indent=2)
            print(f"Saved rejected strategies to {args.rejected_output_file}")

if __name__ == "__main__":
    main()
