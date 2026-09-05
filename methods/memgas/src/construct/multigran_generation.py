import os
import json
from tqdm import tqdm
from openai import OpenAI
import openai
import backoff
import tiktoken

client = OpenAI(
    api_key="",
    base_url="",
)

@backoff.on_exception(backoff.constant, (openai.RateLimitError), 
                      interval=5)
def chat_completions_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)


def summarize_session(entry, model_name, instru_prompt):
    prompt = f"{instru_prompt}\n\n{entry}\n\nYour answer:"
    kwargs = {
        'model': model_name,
        'messages':[
            {"role": "user", "content": prompt}
        ],
        'n': 1,
        'temperature': 0,
        'max_tokens': 500
    }
    completion = chat_completions_with_backoff(client,**kwargs) 
    return completion.choices[0].message.content.strip()


def granularity_generate(dataset,level):
    model_name = 'gpt-4o-mini'
    in_data = json.load(open('../../data/process_data/{dataset}.json'))
    generate_path = '../../multi_granularity_logs/longmemeval-{level}.jsonl'
    
    os.makedirs("../../multi_granularity_logs/", exist_ok=True)
    
    ids2session_text = {}
    for sample in tqdm(in_data):
        conv_id = sample["conversation_id"]
        for sessid, sess in zip(sample['sessions_ids'], sample['sessions']):
            ids2session_text[f'convid-{str(conv_id)}-sessid-{sessid}'] = '\n\n'.join(sess)
            # ids2session_text[sessid] = '\n\n'.join(sess)
    print(len(ids2session_text))
    
    
    if dataset == 'locomo10':
        if level == 'summary_level':
            instru_prompt = "Below is an user-user dialogue memory. Please summarize the following dialogue as concisely as possible in a short paragraph, extracting the main themes and key information.\n"
        elif level == 'keyword_level':
            instru_prompt = "Below is an user-user dialogue memory. Please extract the most relevant keywords, separated by semicolon.\n"
    else:
        if level == 'summary_level':
            instru_prompt = "Below is an user-AI assistant dialogue memory. Please summarize the following dialogue as concisely as possible in a short paragraph, extracting the main themes and key information.\n"
        elif level == 'keyword_level':
            instru_prompt = "Below is an user-AI assistant dialogue memory. Please extract the most relevant keywords, separated by semicolon.\n"


    results = []
    generated_ids = set()
    if os.path.exists(generate_path):
        with open(generate_path, 'r', encoding='utf-8') as file:
            for line in file:
                sample = json.loads(line.strip())
                results.append(sample)
                generated_ids.update(sample.keys())
    
    for ids, entry in ids2session_text.items():
        if ids in generated_ids:
            print(f'generated_ids: {ids}')
            continue
        expansion = summarize_session(entry, model_name, instru_prompt)
        print(ids, expansion)
        results.append({ids:expansion})
    
    with open(generate_path, 'w',encoding='utf-8') as f_write:
        f_write.writelines([json.dumps(_, ensure_ascii=False) + "\n" for _ in results])

if __name__ == '__main__':
    granularity_generate('locomo10','summary_level')
    granularity_generate('locomo10','keyword_level')
    # granularity_generate('LongMTBench+','summary_level')
    # granularity_generate('LongMTBench+','keyword_level')
    # granularity_generate('longmemeval_s','summary_level')
    # granularity_generate('longmemeval_s','keyword_level')
    # granularity_generate('longmemeval_m','summary_level')
    # granularity_generate('longmemeval_m','keyword_level')
    