"""Dataset loader for LoComo evaluation."""
import json
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class QA:
    question: str
    evidence: List[str]
    category: int
    answer: Optional[str] = None
    adversarial_answer: Optional[str] = None


@dataclass
class Turn:
    speaker: str
    dia_id: str
    text: str


@dataclass
class Session:
    date_time: str
    turns: List[Turn]


@dataclass
class Conversation:
    speaker_a: str
    speaker_b: str
    sessions: Dict[str, Session]


@dataclass
class Sample:
    qa: List[QA]
    conversation: Conversation


def load_locomo_dataset(dataset_path: str) -> List[Sample]:
    """Load LoComo dataset from JSON file."""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = []
    for item in data:
        # Parse QA
        qa_list = []
        for qa_item in item['qa']:
            qa = QA(
                question=qa_item['question'],
                evidence=qa_item.get('evidence', []),
                category=qa_item['category'],
                answer=qa_item.get('answer'),
                adversarial_answer=qa_item.get('adversarial_answer')
            )
            qa_list.append(qa)
        
        # Parse conversation
        conv_data = item['conversation']
        sessions = {}
        
        for key, value in conv_data.items():
            if key.startswith('session_') and key.endswith('_date_time'):
                session_num = key.replace('session_', '').replace('_date_time', '')
                session_key = f'session_{session_num}'
                
                if session_key in conv_data and isinstance(conv_data[session_key], list):
                    turns = [
                        Turn(
                            speaker=turn['speaker'],
                            dia_id=turn['dia_id'],
                            text=turn['text']
                        )
                        for turn in conv_data[session_key]
                    ]
                    sessions[session_key] = Session(
                        date_time=value,
                        turns=turns
                    )
        
        conversation = Conversation(
            speaker_a=conv_data['speaker_a'],
            speaker_b=conv_data['speaker_b'],
            sessions=sessions
        )
        
        samples.append(Sample(qa=qa_list, conversation=conversation))
    
    return samples


def load_specific_questions(questions_path: str) -> List[dict]:
    """Load specific questions from JSON file."""
    with open(questions_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    return questions


def filter_dataset_by_questions(samples: List[Sample], specific_questions: List[dict]) -> List[Sample]:
    """Filter dataset to only include specific questions."""
    # Create a set of question texts to look for
    question_texts = {q['question'] for q in specific_questions}
    
    filtered_samples = []
    for sample in samples:
        # Filter QA list to only include specific questions
        filtered_qa = [qa for qa in sample.qa if qa.question in question_texts]
        
        # Only include sample if it has at least one matching question
        if filtered_qa:
            filtered_sample = Sample(
                qa=filtered_qa,
                conversation=sample.conversation
            )
            filtered_samples.append(filtered_sample)
    
    return filtered_samples