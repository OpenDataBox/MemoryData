"""End-to-end test: create NovaMemoryAgent, ingest facts, query, get answer."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from nova_agent import NovaMemoryAgent


def main():
    # 用一个不需要真 API 的 LLM 模型名 — 我们的 agent 会 fallback
    agent = NovaMemoryAgent(
        model="fake-model-for-test",
        retrieve_num=3,
        api_key="sk-fake",
        base_url="http://localhost:9999",
        agent_save_to_folder=str(HERE / "_test_state"),
    )

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

    # 1) Ingest
    print("=" * 60)
    print("Step 1: Ingest", len(facts), "facts")
    for f in facts:
        agent.memorize_chunk(f)
    print(f"  store size: {len(agent.store)}")

    # 2) Recall via MemoryData-compatible API
    print("=" * 60)
    print("Step 2: send_message (memorizing=True) x N")
    for f in facts:
        r = agent.send_message(f, memorizing=True)
        assert r == "", f"memorize 返回应该空: {r}"
    print(f"  store size: {len(agent.store)} (duplicate protection)")

    # 3) Query (LLM fallback - will use chunk as answer)
    print("=" * 60)
    print("Step 3: send_message (memorizing=False) — query + fallback answer")
    queries = [
        "我太太在哪工作?",
        "我开什么车?",
        "我在哪个城市买的房?",
        "我生日是什么时候?",
    ]
    for q in queries:
        chunks = agent.recall_chunks(q)
        ans = agent.send_message(q, memorizing=False)
        print(f"\n  Q: {q}")
        print(f"  Retrieved top-{len(chunks)}:")
        for c in chunks:
            print(f"    - {c[:60]}")
        print(f"  Answer (LLM fallback): {ans[:100]}")

    # 4) Persistence
    print("\n" + "=" * 60)
    print("Step 4: Save/Load round-trip")
    agent.save()
    agent2 = NovaMemoryAgent(
        model="fake", retrieve_num=3,
        agent_save_to_folder=str(HERE / "_test_state"),
    )
    ok = agent2.load()
    assert ok, "load should return True"
    assert len(agent2.store) == len(agent.store), \
        f"size mismatch: {len(agent2.store)} vs {len(agent.store)}"
    # 验证 recall 一致
    for q in queries:
        c1 = set(agent.recall_chunks(q))
        c2 = set(agent2.recall_chunks(q))
        assert c1 == c2, f"recall 不一致: {q}\n  {c1}\n  {c2}"
    print(f"  round-trip ok: {len(agent.store)} chunks preserved")

    # cleanup
    import shutil
    shutil.rmtree(HERE / "_test_state", ignore_errors=True)
    print("\nE2E PASS")


if __name__ == "__main__":
    main()