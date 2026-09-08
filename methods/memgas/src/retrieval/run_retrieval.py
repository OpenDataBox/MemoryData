import os
import multiprocessing as mp
from functools import partial
import json
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import normalize
from eval_utils import evaluate_retrieval
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.insert(0, src_path)

from src.construct.construct_emb import emb_rawdata
from src.construct.construct_asso import construct_asso

def run_ppr(g,reset_prob, damping):
    reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
    pagerank_scores = g.personalized_pagerank(
        # vertices=vertices,
        damping=damping,
        directed=False,
        # weights=g.es["weight"],
        reset=reset_prob,
        implementation='prpack'
    )
    return pagerank_scores
    # return torch.tensor(pagerank_scores).argsort(descending=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    # basic parameters
    parser.add_argument('--retriever', type=str, required=True,)
    parser.add_argument('--method', type=str, required=True)

    parser.add_argument('--num_seednodes', type=int, default=15)
    parser.add_argument('--mem_threshold', type=int, default=30, help="Control the candidate set size.")
    parser.add_argument('--n_components', type=int, default=2, help="Control the number of GMM clustering categories.")
    parser.add_argument('--damping', type=float, default=0.1, help="")
    parser.add_argument('--temp', type=float, default=0.1, help="")
    return parser.parse_args()



def multi_granularity_routing(args, query_emb, granular_embeddings):
    entropies = []
    for emb in granular_embeddings:
        similarity = (query_emb @ emb.T).squeeze()
        prob_dist = F.softmax(similarity / args.temp, dim=0)
        entropy = -torch.sum(prob_dist * torch.log(prob_dist + 1e-12))
        entropies.append(entropy)
    entropies = torch.tensor(entropies)
    # entropies = entropies / torch.sum(entropies)
    # soft_router_weights = (1 - entropies) / sum(1 - entropies) # Shape: (len(granular_embeddings),)

    soft_router_weights = 1 - entropies
    soft_router_weights /= soft_router_weights.sum()
    return soft_router_weights

def main(args):
    if os.path.exists(f'../../data/process_embs/{args.dataset}-{args.retriever}-emb.pt'):
        all_emb = torch.load()
    else:
        all_emb = emb_rawdata(args.dataset, args.retriever)
    
    in_data = json.load(open(f'../../data/process_data/{args.dataset}.json'))
    
    if args.method == 'memgas':
        if os.path.exists(f"../../graph_cache/graph-{args.dataset}-{args.retriever}-{args.mem_threshold}-{args.n_components}.pt"):
            covid2graph = torch.load(f"../../graph_cache/graph-{args.dataset}-{args.retriever}-{args.mem_threshold}-{args.n_components}.pt")
        else:
            covid2graph = construct_asso(args)
    results = []
    for entry, emb in zip(in_data,all_emb):
        assert entry['conversation_id'] == emb['conversation_id']
        
        for qa_one, q_emb in zip(entry['qa'],emb['questions']):
            # correct_docs = list(set( [re.sub(r"-turn_\d+", "", ids) for ids in qa_one['answer_session_ids']] ))
            if args.dataset != "LongMTBench+":
                correct_docs = list(set( [ids for ids in qa_one['answer_session_ids']] ))

            #################### turn emb mean
            turn_num_each_session = [len(sess) for sess in entry['sessions']]
            turn_embeddings = []
            start_idx = 0
            for num_turns in turn_num_each_session:
                if num_turns == 0:
                    turn_mean_emb = torch.zeros(emb['turns'].size(1))
                else:
                    session_turn_embs = emb['turns'][start_idx:start_idx + num_turns]
                    turn_mean_emb = session_turn_embs.mean(dim=0)
                turn_embeddings.append(turn_mean_emb)
                start_idx += num_turns
            turn_embeddings = torch.stack(turn_embeddings)
            ####################
            
            #################### turn-level evaluate single-turn generation
            # correct_docs = list(set(qa_one['answer_session_ids']))
            # scores = (q_emb @ emb['turns'].T).squeeze()
            # rankings = scores.argsort(descending=True)
            # turn_ids = []
            # for sessid, sess in zip(entry['sessions_ids'],entry['sessions']):
            #     for turn_id,turn in enumerate(sess):
            #         turn_ids.append(f"{sessid}-turn_{turn_id + 1}")
            ####################
            
            if args.method == 'session_level':
                scores = (q_emb @ emb['sessions'].T).squeeze()
                rankings = scores.argsort(descending=True)
            elif args.method == 'keyword_level':
                scores = (q_emb @ emb['keywords'].T).squeeze()
                rankings = scores.argsort(descending=True)
            elif args.method == 'summary_level':
                scores = (q_emb @ emb['summarys'].T).squeeze()
                rankings = scores.argsort(descending=True)
            elif args.method == 'hybrid_level':
                scores = (q_emb @ emb['hybrid'].T).squeeze()
                rankings = scores.argsort(descending=True)
            elif args.method == 'turn_level':
                scores = (q_emb @ turn_embeddings.T).squeeze()
                rankings = scores.argsort(descending=True)
            
            elif args.method == 'memgas':
                emb_list = [emb['sessions'], turn_embeddings, emb['summarys'],emb['keywords']]
                soft_router_weights = multi_granularity_routing(args, q_emb, emb_list)
                emb_list = [w * v for w, v in zip(soft_router_weights, emb_list)]
                multi_gran_emb = []
                for i in range(emb['sessions'].size(0)):
                    for e in emb_list:
                        multi_gran_emb.append(e[i])
            
                multi_gran_emb = torch.stack(multi_gran_emb, dim=0)
                scores = (q_emb @ multi_gran_emb.T).squeeze()
                
                topk_values, _ = torch.topk(scores, args.num_seednodes)
                scores[scores < topk_values[-1]] = 0
                scores = run_ppr(covid2graph[entry['conversation_id']], scores, args.damping)
                
                scores = [sum(scores[i:i+4]) for i in range(0, len(scores), 4)]
                rankings = torch.tensor(scores).argsort(descending=True)
            
            cur_results = {
                "conversation_id": entry['conversation_id'],
                'question_type': qa_one['question_type'],
                'question': qa_one['question'],
                'answer': qa_one['answer'],
                'question_date': qa_one['question_date'],
                'retrieval_results': {
                    'ranked_items': [
                        {
                            # 'corpus_id': turn_ids[rid], 
                            'corpus_id': entry['sessions_ids'][rid],
                            'timestamp': entry['sessions_dates'][rid],
                        }
                        for rid in rankings
                    ],
                    'metrics': {
                        'session': {},
                        'turn': {}
                    }
                }
            }
            if args.dataset != "LongMTBench+":
                for k in [1, 3, 5, 10, 30, 50]:
                    recall_any, recall_all, ndcg_any = evaluate_retrieval(rankings, correct_docs, entry['sessions_ids'], k=k)
                    cur_results['retrieval_results']['metrics']['session'].update({
                        'recall_any@{}'.format(k): recall_any,
                        'recall_all@{}'.format(k): recall_all,
                        'ndcg_any@{}'.format(k): ndcg_any
                    })
            results.append(cur_results)
            
    if args.dataset != "LongMTBench+":
        refine_results = []
        for k in results[0]['retrieval_results']['metrics']['session']:
            # will skip abstention instances for reporting the metric
            k_result = np.mean([x['retrieval_results']['metrics']['session'][k] for x in results if '_abs' not in str(x['conversation_id'])])
            if k.startswith("recall_all@") or k.startswith("ndcg_any@"):
                refine_results.append(f"{round(k_result*100, 2)}")
        print("\t".join(refine_results))
    

    # save results
    os.makedirs("../../retrieval_logs/", exist_ok=True)
    out_file=f"../../retrieval_logs/{args.dataset}-{args.retriever}-{args.method}.jsonl"
    out_f = open(out_file, 'w')
    for entry in results:
        print(json.dumps(entry), file=out_f)
    out_f.close()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)
