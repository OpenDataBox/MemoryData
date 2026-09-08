
import json
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.insert(0, src_path)
from generation.metrics import evaluate_match, evaluate_sim
from tqdm import tqdm
from collections import defaultdict
import argparse

def get_qa_metrics(results):
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

    pred_all = []
    for res in results:
        ans = get_answer(res["response"])
        pred_all.append(ans)

    answer_all = []
    for res in results:
        answer_all.append(str(res["answer"]))

    metrics = evaluate_sim(pred_all, answer_all, truncate_pred=False)
    metrics.update(evaluate_match(pred_all, answer_all, truncate_pred=False))
    metrics = {key: round(value * 100, 2) for key, value in metrics.items()}
    return metrics

def calculate_single(results):
    results_acc = []
    for sample in results:
            llm_j = sample['llm_judge_single']
            if '[[yes]]' in llm_j and '[[no]]' not in llm_j:
                results_acc.append(1)
            else:
                results_acc.append(0)
    return round(sum(results_acc)/len(results_acc)*100,2)


# locomo10 longmemeval_s longmemeval_m LongMTBench+
# {1, 2, 3, 4, 5}
# {'single-session-preference', 'single-session-user', 'knowledge-update', 'temporal-reasoning', 'multi-session', 'single-session-assistant'}


def print_type_metric(save_path):
    locomo10_qt = {
        1:'multi-hop retrieval',
        2:'temporal reasoning',
        3:'open domain knowledge',
        4:'single-hop retrieval',
        5:'adversarial',
    }
    type_results = defaultdict(list)
    with open(save_path, "r") as f:
        for line in f.readlines():
            sample = json.loads(line.strip())
            if 'locomo10' in  save_path:
                type_results[locomo10_qt[sample['question_type']]].append(sample)
            else:
                type_results[sample['question_type']].append(sample)
                
    type_metrics = {}
    for Type, results in type_results.items():
        qa_metrics = get_qa_metrics(results)
        # gpt_judge = calculate_single(results)
        # qa_metrics['GPT-4o-J'] = gpt_judge
        type_metrics[Type] = qa_metrics
    print('-------------------')
    print(save_path)
    print(type_metrics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="long-term conversation evaluation")
    parser.add_argument('--eval_file', type=str, required=True)
    args = parser.parse_args()
    print_type_metric(args.eval_file)

