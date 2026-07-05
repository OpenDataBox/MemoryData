"""Mock MemoryAgentBench dataset — for testing runner offline.

Mirrors the structure of ai-hyz/MemoryAgentBench Accurate_Retrieval split:
each sample has {metadata, context_chunks, qa_list}.

10 mock samples covering: name, car, family, pet, birthday, job-hopping,
housing, programming, car, hobbies.
"""
from typing import List, Dict, Any


MOCK_SAMPLES: List[Dict[str, Any]] = [
    {
        "metadata": {"id": "mock_001", "source": "eventqa_full"},
        "context_chunks": [
            "用户的名字是张伟,职业是工程师,在上海工作。",
            "用户的太太叫李娜,也是工程师,在阿里巴巴工作。",
            "用户养了一只猫叫橘子,橘色短毛,3 岁。",
            "用户最近换工作,从字节跳动跳槽到腾讯。",
            "用户生日 1990 年 3 月 15 日。",
        ],
        "qa_list": [
            {"question": "用户的名字是什么?", "answers": ["张伟"]},
            {"question": "用户太太叫什么名字?", "answers": ["李娜"]},
            {"question": "用户的猫叫什么?", "answers": ["橘子"]},
            {"question": "用户的生日是哪一天?", "answers": ["1990年3月15日", "3月15日"]},
            {"question": "用户最近从哪家公司跳槽到哪家公司?", "answers": ["字节跳动到腾讯", "字节跳动", "腾讯"]},
        ],
    },
    {
        "metadata": {"id": "mock_002", "source": "eventqa_full"},
        "context_chunks": [
            "用户2025年在杭州买房,花费300万。",
            "用户的儿子2024年出生。",
            "用户最近在学习Rust编程语言。",
            "用户的车是特斯拉Model Y,2023款。",
            "用户喜欢打篮球和跑步。",
        ],
        "qa_list": [
            {"question": "用户在哪一年买房?", "answers": ["2025年", "2025"]},
            {"question": "用户在哪个城市买的房?", "answers": ["杭州"]},
            {"question": "用户买房花了多少钱?", "answers": ["300万"]},
            {"question": "用户最近在学习什么编程语言?", "answers": ["Rust"]},
            {"question": "用户开什么车?", "answers": ["特斯拉Model Y", "Model Y"]},
        ],
    },
    {
        "metadata": {"id": "mock_003", "source": "eventqa_full"},
        "context_chunks": [
            "陈先生是一位资深产品经理,目前在字节跳动工作。",
            "他的爱好是摄影和徒步旅行。",
            "他养了一只金毛犬叫豆豆。",
            "他2018年在北京买了房,首付200万。",
            "他的太太王女士是设计师,在家工作。",
        ],
        "qa_list": [
            {"question": "陈先生在哪里工作?", "answers": ["字节跳动"]},
            {"question": "陈先生的爱好是什么?", "answers": ["摄影和徒步", "摄影", "徒步"]},
            {"question": "陈先生养的狗叫什么?", "answers": ["豆豆"]},
            {"question": "陈先生在哪一年买的房?", "answers": ["2018年", "2018"]},
            {"question": "陈先生的太太是做什么的?", "answers": ["设计师"]},
        ],
    },
]


def get_mock_samples() -> List[Dict[str, Any]]:
    """Returns 3 mock samples, total 15 QA pairs across Chinese scenarios."""
    return MOCK_SAMPLES