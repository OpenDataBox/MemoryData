
import argparse
import json
import os
from tqdm import tqdm
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.insert(0, src_path)
import asyncio
from generation.async_llm import run_async

PROMPT_G = """
I will give you a question, a reference answer, and a response from a model. Please answer [[yes]] if the response contains the reference answer. Otherwise, answer [[no]]. 
If the response is equivalent to the correct answer or contains all the intermediate steps to get the reference answer, you should also answer [[yes]]. If the response only contains a subset of the information required by the answer, answer [[no]]. 

[User Question]
{question}

[The Start of Reference Answer]
{answer}
[The End of Reference Answer]

[The Start of Model’s Response]
{response}
[The End of Model’s Response]

Is the model response correct? Answer [[yes]] or [[no]] only.
"""

# locomo10 longmemeval_s longmemeval_m LongMTBench+
# memgas_filter
# python llm_judge_single.py --model_name_or_path gpt-4o --eval_file  longmemeval_s-contriever-memgas_filter-gpt-4o-mini-topk_3.jsonl


parser = argparse.ArgumentParser(description="long-term conversation evaluation")
parser.add_argument('--eval_file', type=str, required=True)
parser.add_argument("--model_name_or_path", type=str, default="gpt-4o")
args = parser.parse_args()

g_path = f'../../generation_logs/{args.eval_file}'

file_name = os.path.basename(g_path)
save_path = f'../../generation_logs/llm_judge_single/{file_name}-{args.model_name_or_path}_judge.jsonl'

def calculate_single(path):
    results = []
    with open(path, "r") as f:
        for line in f.readlines():
            sample = json.loads(line.strip())
            llm_j = sample['llm_judge_single']
            if '[[yes]]' in llm_j and '[[no]]' not in llm_j:
                results.append(1)
            else:
                results.append(0)
    print("llm_judge_single Acc: ",round(sum(results)/len(results)*100,2))

if os.path.exists(save_path):
    calculate_single(save_path)
else:
    g_results = []
    async_prompts = []
    with open(g_path, "r") as f:
        for line in f.readlines():
            sample = json.loads(line.strip())
            g_results.append(sample)
            prompt = PROMPT_G.format(question=sample['question'],answer=sample['answer'],response=sample['response'])
            async_prompts.append(prompt)

    eval_results = []
    async_responses = asyncio.run(run_async(async_prompts,args.model_name_or_path))
    for sample, response in zip(g_results,async_responses):
        sample["llm_judge_single"] = response
        sample.pop("retrieval_results", None)
        eval_results.append(sample)

    with open(save_path, "w", encoding="utf-8") as f:
        f.writelines([json.dumps(_, ensure_ascii=False) + "\n" for _ in eval_results])
    calculate_single(save_path)
