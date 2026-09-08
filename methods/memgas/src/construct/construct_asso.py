import torch
import numpy as np
import igraph as ig
from tqdm import tqdm
import json
from sklearn.mixture import GaussianMixture
import argparse
import os
from construct_emb import emb_rawdata

def gmm_edge(sim_scores,mem_threshold,n_components):
    sim_scores = sim_scores.numpy()
    sorted_indices = np.argsort(sim_scores)[::-1] 
    n_candidates = min(mem_threshold, len(sim_scores))
    top_candidate_indices = sorted_indices[:n_candidates]
    
    candidate_scores = sim_scores[top_candidate_indices].reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(candidate_scores)
    labels = gmm.predict(candidate_scores)
    means = gmm.means_.flatten()
    higher_class = np.argmax(means)
    
    selected_mask = (labels == higher_class)
    selected_indices = top_candidate_indices[selected_mask]
    
    return selected_indices

def get_edges(cur_mem, cur_mem_idx, pre_mem, mem_threshold, n_components):
    
    pre_mem = torch.stack(pre_mem)
    # norms = torch.norm(pre_mem, dim=1, keepdim=True)  # Compute L2 norm
    # normalized_embs = torch.where(norms > 0, pre_mem / norms, torch.zeros_like(pre_mem))
    sim_scores = cur_mem @ pre_mem.T

    edges = []
    selected_indices = gmm_edge(sim_scores, mem_threshold, n_components)
    ######## topk edge
    # top_k = 5
    # selected_indices = torch.topk(sim_scores, k=min(top_k, len(pre_mem)), largest=True).indices.tolist()
    ########
    for idx in selected_indices:
        edges.append((cur_mem_idx, idx))  # Add edge with type
    return edges

def construct_asso(args):    
    
    in_data = json.load(open(f'../../data/process_data/{args.dataset}.json'))
    all_emb = torch.load(f'../../data/process_embs/{args.dataset}-{args.retriever}-emb.pt')
    if os.path.exists(f'../../data/process_embs/{args.dataset}-{args.retriever}-emb.pt'):
        all_emb = torch.load()
    else:
        all_emb = emb_rawdata(args.dataset, args.retriever)

    covid2graph = {}
    for entry, emb in tqdm(zip(in_data,all_emb), total=min(len(in_data), len(all_emb))):
        assert entry['conversation_id'] == emb['conversation_id']
        
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

        node_ids = []
        for sess_id in entry['sessions_ids']:
            node_ids.append(f"{sess_id}-session")
            node_ids.append(f"{sess_id}-turn")
            node_ids.append(f"{sess_id}-summary")
            node_ids.append(f"{sess_id}-keyword")

        node_embs = []
        for i in range(len(entry['sessions_ids'])):
            node_embs.append(emb['sessions'][i])
            node_embs.append(turn_embeddings[i])
            node_embs.append(emb['summarys'][i])
            node_embs.append(emb['keywords'][i])
        
        all_edges = []
        for i in range(5*4,len(node_ids)):
            cur_mem = node_embs[i]
            pre_mem = node_embs[:i // 4 * 4]
            edges = get_edges(cur_mem, i, pre_mem, args.mem_threshold, args.n_components)
            all_edges += edges
        edge_tuples = [(edge[0], edge[1]) for edge in all_edges]  # Extract edge tuples
        n_vertices = len(node_ids)
        g = ig.Graph(n_vertices, edge_tuples, directed=False)
        g["conversation_id"] = entry['conversation_id']
        g.vs["name"] = node_ids
        
        covid2graph[entry['conversation_id']] = g
    
    os.makedirs("../../graph_cache", exist_ok=True)
    torch.save(covid2graph,f"../../graph_cache/graph-{args.dataset}-{args.retriever}-{args.mem_threshold}-{args.n_components}.pt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--retriever', type=str, required=True,)
    parser.add_argument('--mem_threshold', type=int, default=20, help="Control the candidate set size.")
    parser.add_argument('--n_components', type=int, default=2, help="Control the number of GMM clustering categories.")
    args = parser.parse_args()
    construct_asso(args)