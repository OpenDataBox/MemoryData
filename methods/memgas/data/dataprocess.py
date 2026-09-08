import os
from tqdm import tqdm
import json

def process_longmemeval():
    ## longmemeval
    # "question_id": "gpt4_2655b836",
    # "question_type": "temporal-reasoning",
    # "question": "What was the first issue I had with my new car after its first service?",
    # "answer": "GPS system not functioning correctly",
    # "question_date": "2023/04/10 (Mon) 23:07",
    # "haystack_dates": [
    # "haystack_session_ids": [
    # "haystack_sessions": [
    # "answer_session_ids": [

    in_data = json.load(open('origin_data/longmemeval_s'))
    alldata = []

    for entry in tqdm(in_data):
        question_id = entry['question_id']
        question_type = entry['question_type']
        question = entry['question']
        answer = entry['answer']
        question_date = entry['question_date']
        haystack_dates = entry['haystack_dates']
        haystack_session_ids = entry['haystack_session_ids']
        haystack_sessions = entry['haystack_sessions']
        answer_session_ids = []
        for cur_sess_id, sess_entry, ts in zip(entry['haystack_session_ids'], entry['haystack_sessions'], entry['haystack_dates']):
            for turn_id, turn in enumerate(sess_entry):
                if 'has_answer' in turn and turn['has_answer']==True:
                    # answer_session_ids.append(f"{cur_sess_id.replace('answer_','')}-turn_{turn_id // 2 + 1}")
                    answer_session_ids.append(f"{cur_sess_id.replace('answer_','')}")

        sessions = []
        for sess_entry in entry['haystack_sessions']:
            # new_session = [{k: v for k, v in item.items() if k != "has_answer"} for item in sess_entry]
            session = []
            for item in sess_entry:
                session.append(f"[{item['role']}]: {item['content']}")
            merged_session = []
            for i in range(0, len(session), 2):
                if i + 1 < len(session):
                    merged_session.append(session[i] + "\n" + session[i+1])
                else:
                    merged_session.append(session[i])
            if len(merged_session)==0:
                print(len(merged_session),len(session))
            sessions.append(merged_session)
            
        dataform = {
            'conversation_id':entry['question_id'],
            'qa':[
                {
                "question": entry['question'],
                "question_type": entry['question_type'],
                "question_date":entry['question_date'],
                "answer": entry['answer'],
                "answer_session_ids": answer_session_ids,
            },
            ],
            'sessions_ids':[s.replace('answer_', '') for s in entry['haystack_session_ids']],
            'sessions_dates':entry['haystack_dates'],
            'sessions':sessions
            }
        alldata.append(dataform)
        
    with open("./process_data/longmemeval_s.json", "w", encoding="utf-8") as f:
        json.dump(alldata, f, ensure_ascii=False, indent=4)

def process_locomo10():
    ## locomo10
    from tqdm import tqdm
    import json
    in_data = json.load(open('origin_data/locomo10.json'))

    alldata = []

    for entry in tqdm(in_data):
        # dict_keys(['qa', 'conversation', 'event_summary', 'observation', 'session_summary', 'sample_id'])
        
        
        newqa = []
        for qaitem in entry['qa']:
            if 'adversarial_answer' in qaitem:
                answer = qaitem['adversarial_answer']
            else:
                answer = qaitem['answer']
            answer_session_ids = []
            for item in qaitem['evidence']:
                try:
                    turn_id = int(item.split(':')[1])
                except:
                    continue
                # answer_session_ids.append(f"{item.replace('D', 'session_').split(':')[0]}-turn_{turn_id  // 2 + 1}")
                answer_session_ids.append(f"{item.replace('D', 'session_').split(':')[0]}")

            newqa.append(
                {
                "question": qaitem['question'],
                "question_type": qaitem['category'],
                "question_date": None,
                "answer": answer,
                "answer_session_ids":answer_session_ids,
            })
            
        conversation = entry['conversation']
        sessions_ids = []
        sessions_dates = []
        sessions = []
        for i in range(1000):
            if f'session_{i+1}' in conversation:
                sessions_ids.append(f'session_{i+1}')
                sessions_dates.append(conversation[f'session_{i+1}_date_time'])
                session = []
                for dialog in conversation[f'session_{i+1}']:
                    # 替换 speaker -> role 和 text -> content
                    if 'blip_caption' in dialog:
                        session.append(f"[{dialog['speaker']}]: {dialog['text']}\n The image Caption: {dialog['blip_caption']}")
                    else:
                        session.append(f"[{dialog['speaker']}]: {dialog['text']}")
                merged_session = []
                for i in range(0, len(session), 2):
                    if i + 1 < len(session):
                        merged_session.append(session[i] + "\n" + session[i+1])
                    else:
                        merged_session.append(session[i])

                sessions.append(merged_session)
        dataform = {
            'conversation_id':entry['sample_id'],
            'qa':newqa,
            'sessions_ids':sessions_ids,
            'sessions_dates':sessions_dates,
            'sessions':sessions
            }
        alldata.append(dataform)
        

    with open("./process_data/locomo10.json", "w", encoding="utf-8") as f:
        json.dump(alldata, f, ensure_ascii=False, indent=4)
        
def process_LongMTBench():
    ## Long-MT-Bench+
    from tqdm import tqdm
    import json
    in_data = json.load(open('origin_data/Long-MT-Bench-Plus.json'))
    alldata = []
    for entry in tqdm(in_data):
        # print(entry.keys()) #dict_keys(['sessions', 'questions', 'conversation_id', 'turns', 'answers'])
        newqa = []
        for q, a in zip(entry['questions'],entry['answers']):
            newqa.append(
                {
                "question": q,
                "question_type": None,
                "question_date": None,
                "answer": a,
                "answer_session_ids": None,
            })
        sessions_ids = []
        sessions_dates = []
        for i, session in enumerate(entry['sessions']):
            sessions_ids.append(f'session_{i+1}')
            sessions_dates.append(None)
        dataform = {
            'conversation_id':entry['conversation_id'],
            'qa':newqa,
            'sessions_ids':sessions_ids,
            'sessions_dates':sessions_dates,
            'sessions':entry['sessions']
            }
        alldata.append(dataform)
    with open("./process_data/Long-MT-Bench-Plus.json", "w", encoding="utf-8") as f:
        json.dump(alldata, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    os.makedirs("./process_data/", exist_ok=True)
    process_longmemeval()
    process_locomo10()
    process_LongMTBench()