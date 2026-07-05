"""Smoke test nova_core against nova-mvp's run_memory_tests.py scenarios."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nova_core import (
    expand_morph, tokenize, NovaMemoryStore, _MORPH_MAP,
)


def assert_eq(label, actual, expected):
    if actual != expected:
        raise AssertionError(f"{label}: 期望 {expected!r}, 实际 {actual!r}")
    print(f"  PASS {label}")


def assert_true(label, cond):
    if not cond:
        raise AssertionError(f"{label}: FAILED")
    print(f"  PASS {label}")


def main():
    # ---- MORPH
    assert_eq("morph len>=30", len(_MORPH_MAP) >= 30, True)
    assert_eq("morph 买的房", expand_morph("买的房"), "买房")
    assert_eq("morph 在哪工作", expand_morph("在哪工作"), "工作")
    assert_eq("morph 开什么车", expand_morph("开什么车"), "车")
    assert_eq("morph 几口人", expand_morph("几口人"), "家庭成员")
    assert_eq("morph 哪个城市", expand_morph("哪个城市"), "城市")
    assert_eq("morph 不改原句", expand_morph("我今天很开心"), "我今天很开心")

    # ---- TOKENIZE
    assert_true("tokenize 买的房→买房",
                "买房" in tokenize("我在哪个城市买的房,花了多少钱?"))
    assert_true("tokenize 城市保留",
                "城市" in tokenize("我在哪个城市买的房,花了多少钱?"))
    assert_true("tokenize 单字 车", "车" in tokenize("我开什么车?"))
    assert_true("tokenize 单字 猫", "猫" in tokenize("我的猫叫什么?"))
    assert_true("tokenize 英文 model",
                "model" in tokenize("I drive a Tesla Model Y."))
    assert_eq("tokenize 空字符串", tokenize(""), [])
    assert_eq("tokenize None", tokenize(None), [])
    assert_true("tokenize 纯停用词 不崩溃",
                isinstance(tokenize("的了吗呢啊"), list))

    # ---- STORE: 10-question benchmark
    store = NovaMemoryStore()
    facts = [
        "用户的名字是张伟,职业是工程师,在上海工作",
        "用户的太太叫李娜,是工程师,在阿里巴巴工作",
        "用户养了一只猫叫橘子,橘色短毛,3 岁",
        "用户最近换工作,从字节跳动跳槽到腾讯",
        "用户生日 1990 年 3 月 15 日",
        "用户 2025 年在杭州买房,花费 300 万",
        "用户的儿子 2024 年出生",
        "用户最近在学习 Rust 编程语言",
        "用户的车是特斯拉 Model Y,2023 款",
        "用户喜欢打篮球和跑步",
    ]
    store.memorize(facts, keywords=facts)  # 把 content 同时做 keywords

    queries = [
        ("我太太在哪工作?", ["阿里巴巴"]),
        ("我叫什么名字?", ["张伟"]),
        ("我的猫叫什么?", ["橘子"]),
        ("我生日是什么时候?", ["1990", "3 月 15"]),
        ("我开什么车?", ["特斯拉", "Model Y"]),
        ("我之前在哪工作,现在在哪?", ["字节", "腾讯"]),
        ("我太太和我是什么关系,我们都做什么工作?", ["太太", "工程师"]),
        ("我家有几口人,各自多大?", ["张伟", "李娜", "儿子"]),
        ("我在哪个城市买的房,花了多少钱?", ["杭州", "300"]),
        ("我最近在学的编程语言,和我的车是什么关系?", ["Rust", "特斯拉"]),
    ]
    hits = 0
    for q, exp in queries:
        results = store.recall(q, k=5)
        joined = "\n".join(c for c, _ in results)
        if any(e in joined for e in exp):
            hits += 1
        else:
            print(f"  FAIL Q: {q}")
            print(f"    expected one of: {exp}")
            for c, s in results[:3]:
                print(f"    score={s} -> {c[:60]}")

    assert_eq("benchmark 10 题 (got N/10)", hits, 10)
    print("\n全部测试通过 OK")


if __name__ == "__main__":
    main()