import json
from methods.retentionmem.datasets import load_cases


def test_caption_is_observation_and_adversarial_answer_is_distractor(tmp_path):
    data = [{"sample_id": "s", "conversation": {"session_1": [{"dia_id": "D1:1", "speaker": "A", "text": "Here it is", "blip_caption": "a blue kayak"}]},
             "qa": [{"question": "Unsupported", "category": 5, "adversarial_answer": "wrong distractor", "evidence": []}]}]
    path = tmp_path/"dataset.json"; path.write_text(json.dumps(data))
    primary = list(load_cases(path, "locomo"))[0]
    extended = list(load_cases(path, "locomo", True))[0]
    assert "blue kayak" in primary.events[0].text and len(primary.questions) == 0
    assert extended.questions[0].answer == "No information available"
    assert "wrong distractor" not in str(extended)
