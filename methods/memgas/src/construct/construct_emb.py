import json


from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm
import json
from sentence_transformers import SentenceTransformer
import os
# embform = [
#     {
#         "conversation_id": 0,
#         "questions": tensor,
#         "sessions": tensor,
#         "turns": tensor,
#     },
# ]


def read_ids2granularity(path):
    ids2granularity = {}
    #
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())  
            ids2granularity.update(data)
    return ids2granularity

class EmbeddingModelContriever():
    def __init__(self):
        # if self.args.retriever == 'flat-contriever':
        self.model = AutoModel.from_pretrained('facebook/contriever').to(torch.device('cuda', 0))
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')

    def get_emb_contriever(self, expansion_ids, expansion):
        def mean_pooling(token_embeddings, mask):
            token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
            sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
            return sentence_embeddings
        
        with torch.no_grad():
            all_docs_vectors = []
            dataloader = DataLoader(expansion, batch_size=64, shuffle=False)
            for batch in tqdm(dataloader):
                inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                cur_docs_vectors = mean_pooling(outputs[0], inputs['attention_mask']).detach().cpu()
                all_docs_vectors.append(cur_docs_vectors)
            all_docs_vectors = torch.concat(all_docs_vectors, axis=0)
        
        if expansion_ids:
            ids2emb = {}
            for i in range(len(expansion_ids)):
                ids2emb[expansion_ids[i]] = all_docs_vectors[i]
            return ids2emb
        else:
            return all_docs_vectors

class EmbeddingModelSBERT():
    def __init__(self, retriever):
        if retriever == 'mpnet':
            self.model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-cos-v1')
        elif retriever == 'minilm':
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        elif retriever == 'qaminilm':
            self.model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
    def get_emb_contriever(self, expansion_ids, expansion):
        all_docs_vectors = self.model.encode(expansion)
        if expansion_ids:
            ids2emb = {}
            for i in range(len(expansion_ids)):
                ids2emb[expansion_ids[i]] = all_docs_vectors[i]
            return ids2emb
        else:
            return torch.tensor(all_docs_vectors)


def emb_rawdata(dataset, retriever):
    os.makedirs('../../data/process_embs/', exist_ok=True)
    data_path=f'../../data/process_data/{dataset}.json'
    save_path=f'../../data/process_embs/{dataset}-{retriever}-emb.pt'
    in_data = json.load(open(data_path))
    if 'longmemeval' in dataset:
        dataset = 'longmemeval'
    ids2summary = read_ids2granularity(f'../../multi_granularity_logs/{dataset}-summary_level.jsonl')
    ids2keyword = read_ids2granularity(f'../../multi_granularity_logs/{dataset}-keyword_level.jsonl')
    
    all_emb = []
    if retriever == 'contriever':
        emb_model = EmbeddingModelContriever()
    elif retriever == 'mpnet' or retriever == 'minilm' or retriever == 'qaminilm':
        emb_model = EmbeddingModelSBERT(retriever)
    for conversation in tqdm(in_data):
        questions = [qa_item["question"] for qa_item in conversation["qa"]]
        sessions = ['\n'.join(session) for session in conversation["sessions"]]
        turns = []
        for session in conversation["sessions"]:
            for turn in session:
                turns.append(turn)

        summarys = []
        keywords = []
        for sessid in conversation["sessions_ids"]:
            if 'longmemeval' in dataset:
                id = sessid
            else:
                id = f"convid-{str(conversation['conversation_id'])}-sessid-{sessid}"
            summarys.append(str(ids2summary[id]))
            keywords.append(str(ids2keyword[id]))
        
        hybrid = [f"{summary} {keyword} {session}" for session, summary, keyword in zip(sessions, summarys, keywords)]

        print(len(questions),len(sessions),len(turns),len(summarys),len(keywords),len(hybrid))
        questions = emb_model.get_emb_contriever(None, questions)
        sessions = emb_model.get_emb_contriever(None, sessions)
        turns = emb_model.get_emb_contriever(None, turns)
        summarys = emb_model.get_emb_contriever(None, summarys)
        keywords = emb_model.get_emb_contriever(None, keywords)
        hybrid = emb_model.get_emb_contriever(None, hybrid)
        embform = {
                "conversation_id": conversation['conversation_id'],
                "questions": questions,
                "sessions": sessions,
                "turns": turns,
                "summarys": summarys,
                "keywords": keywords,
                "hybrid": hybrid,
            }
        all_emb.append(embform)
    torch.save(all_emb,save_path)
    

if __name__ == '__main__':
    emb_rawdata('LongMTBench+', 'contriever')
    emb_rawdata('locomo10', 'contriever')
    emb_rawdata('longmemeval_s', 'contriever')
    emb_rawdata('longmemeval_m', 'contriever')
