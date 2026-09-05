
import argparse
import json
import os

from metrics import evaluate_match, evaluate_sim
from tqdm import tqdm
from utils import LocalLLM, OpenAILLM
from async_llm import run_async
import asyncio
import aiohttp

PROMPT_G = """
You are an intelligent dialog bot. You will be shown History Dialogs. Please read, memorize, and understand the given Dialogs, then generate one concise, coherent and helpful response for the Question.

History Dialogs: {retrieved_texts}

Question Date: {question_date}
Question: {question}
"""
# locomo10 longmemeval_s longmemeval_m LongMTBench+
# python generation_baselines.py --dataset longmemeval_s --retriever contriever --model_name_or_path gpt-4o-mini --topk 3 --method session_level


parser = argparse.ArgumentParser(description="long-term conversation evaluation")
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--retriever', type=str, required=True)
parser.add_argument("--model_name_or_path", type=str, default="gpt-4o-mini")
parser.add_argument('--topk', type=int, required=True)
parser.add_argument('--method', type=str, required=True)

args = parser.parse_args()

save_path = f'../../generation_logs/{args.dataset}-{args.retriever}-{args.method}-{args.model_name_or_path}-topk_{args.topk}.jsonl'
os.makedirs(os.path.dirname(save_path), exist_ok=True)


in_data = json.load(open(f'../../data/process_data/{args.dataset}.json'))




# results = []
# processed_query = set()
# if os.path.exists(save_path):
#     with open(save_path, "r") as f:
#         for line in f.readlines():
#             sample = json.loads(line.strip())
#             results.append(sample)
#     for sample in results:
#         processed_query.add(sample["question"])


################ session based
conv2sessions = {}
for entry in in_data:
    ids2session = {k:v for k,v in zip(entry['sessions_ids'],entry['sessions'])}
    conv2sessions[entry['conversation_id']] = ids2session
################

################ other granularity
# def get_multigran(level):
#     conv2multigran = {}
#     summ_dict = {}
#     if 'longmemeval' in args.dataset:
#         multi_gran_path = f'../../multi_granularity_logs/longmemeval-{level}.jsonl'
#     else:
#         multi_gran_path = f'../../multi_granularity_logs/{args.dataset}-{level}.jsonl'
#     with open(multi_gran_path, "r") as f:
#         for line in f.readlines():
#             summ_dict.update(json.loads(line.strip()))
#     for entry in in_data:
#         ids2summary = {}
#         conv_id = entry['conversation_id']
#         for sessid in entry['sessions_ids']:
#             if 'longmemeval' in args.dataset:
#                 summ_text = summ_dict[f'{sessid}']
#             else:
#                 summ_text = summ_dict[f'convid-{str(conv_id)}-sessid-{sessid}']
#             ids2summary[sessid] = summ_text
#         conv2multigran[conv_id] = ids2summary
#     return conv2multigran
# conv2summary = get_multigran('summary_level')
# conv2keyword = get_multigran('keyword_level')
########################


######## 并行 
retrieved_data = []
with open(f'../../retrieval_logs/{args.dataset}-{args.retriever}-{args.method}.jsonl', "r") as f:
    for line in f.readlines():
        retrieved_data.append(json.loads(line.strip()))

async_prompts = []
for idx, sample in enumerate(tqdm(retrieved_data)):
    ids2session = conv2sessions[sample["conversation_id"]]
    retrieved_texts = ""
    for retrieved_sess in sample['retrieval_results']['ranked_items'][:args.topk]:
        ######## turn_level ########
        # ssesid, turnid = retrieved_sess['corpus_id'].split('-turn_')
        # session = ids2session[ssesid]
        # session = session[int(turnid)-1]
        # retrieved_texts += f"\n### Session Content:\n{session}\n"
        ############################
        session = ids2session[retrieved_sess['corpus_id']]
        retrieved_texts += f"\n### Session Date: {retrieved_sess['timestamp']}\nSession Content:\n{session}\n"
    prompt = PROMPT_G.format(retrieved_texts=retrieved_texts, question=sample["question"], question_date=sample["question_date"])
    async_prompts.append(prompt)

results = []
async_responses = asyncio.run(run_async(async_prompts))
for sample, response in zip(retrieved_data, async_responses):
    sample["response"] = response
    results.append(sample)
with open(save_path, "w", encoding="utf-8") as f:
    f.writelines([json.dumps(_, ensure_ascii=False) + "\n" for _ in results])


def get_answer(ans):
    strip_word_list = [
        "\nDialogs:",
        "\n[bot]:",
        "\nAssistant:",
        "\nReview:",
        "\n",
        "[bot]:",
    ]
    cut_word_list = ["\n[human]:", "\nQuestion:", "\nQ:"]

    for strip_word in strip_word_list:
        if ans is not None:
            ans = ans.strip(strip_word)
        else:
            ans = ""
        
    for cut_word in cut_word_list:
        if cut_word in ans:
            ans = ans.split(cut_word)[0]
    return ans

print('Calculating metrics')
pred_all = []
for res in results:
    ans = get_answer(res["response"])
    pred_all.append(ans)

answer_all = []
for res in results:
    answer_all.append(str(res["answer"]))

metrics = evaluate_sim(pred_all, answer_all, truncate_pred=False)
metrics.update(evaluate_match(pred_all, answer_all, truncate_pred=False))
print(metrics)
metrics_dir = os.path.join(os.path.dirname(save_path), "metrics")
os.makedirs(metrics_dir, exist_ok=True)
with open(
    os.path.join(
        metrics_dir, os.path.basename(save_path).replace("answer", "metrics")
    ),
    "w",
    encoding="utf-8",
) as f:
    json.dump(metrics, f)
