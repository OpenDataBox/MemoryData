"""Nova Memory core — vendored from nova-mvp/memory.py (recall primitives only).

Zero external deps. Drop into MemoryData as a lexical + morphology-only
memory baseline.
"""
from __future__ import annotations

import re
from typing import List, Dict, Tuple, Set

# ============================================================
# 完全复制自 nova-mvp/memory.py
# ============================================================
_TOKEN_RE = re.compile(r"[\w一-鿿]+")

_STOPWORDS = {
    "的","了","和","是","在","我","你","他","她","它","也","就","都","要","会","能","可以","怎么",
    "什么","哪","这","那","这个","那个","一个","一下","下","吗","呢","啊","吧","呀","哦","嗯",
    "a","an","the","is","are","was","were","be","been","being","do","does","did","have","has","had",
    "i","you","he","she","it","we","they","what","how","why","when","where","who",
}

_MORPH_MAP = {
    "买的房": "买房", "买的房子": "买房", "买的车": "买车",
    "买的猫": "买猫", "买的狗": "买狗",
    "租的房": "租房", "租的房子": "租房",
    "在哪工作": "工作", "在哪上班": "工作", "在哪里工作": "工作",
    "在哪儿工作": "工作", "干啥工作": "工作",
    "叫什么": "叫", "叫什么名字": "叫", "叫啥": "叫", "叫啥名字": "叫",
    "什么时候生日": "生日", "哪天生日": "生日", "生日是什么时候": "生日",
    "几口人": "家庭成员", "有谁": "家庭成员", "家里有谁": "家庭成员", "家里几个人": "家庭成员",
    "在学什么": "学", "学习什么": "学", "在学啥": "学",
    "喜欢什么": "爱好", "爱好是": "爱好", "喜欢干啥": "爱好", "喜欢玩啥": "爱好",
    "之前在哪工作": "跳槽", "之前干啥的": "跳槽",
    "开的什么车": "车", "开的啥车": "车", "开什么车": "车",
    "哪个城市": "城市", "在哪个城市": "城市", "哪座城市": "城市",
    "在哪儿": "城市", "在哪个地方": "城市",
}


def expand_morph(text: str) -> str:
    if not text:
        return text
    out = text
    for k, v in _MORPH_MAP.items():
        if k in out:
            out = out.replace(k, v)
    return out


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    out: List[str] = []
    seen: Set[str] = set()

    raw = _TOKEN_RE.findall(text.lower())
    for w in raw:
        if w in _STOPWORDS or len(w) < 1:
            continue
        if all('一' <= ch <= '鿿' for ch in w) and len(w) > 3:
            for n in (2, 3):
                for i in range(len(w) - n + 1):
                    s = w[i:i + n]
                    if s not in seen:
                        seen.add(s); out.append(s)
        else:
            if w not in seen:
                seen.add(w); out.append(w)

    norm_tokens: List[str] = []
    text_norm = expand_morph(text)
    if text_norm != text:
        raw_norm = _TOKEN_RE.findall(text_norm.lower())
        for w in raw_norm:
            if w in _STOPWORDS or len(w) < 1:
                continue
            if all('一' <= ch <= '鿿' for ch in w) and len(w) > 3:
                for n in (2, 3):
                    for i in range(len(w) - n + 1):
                        s = w[i:i + n]
                        if s not in seen:
                            seen.add(s); norm_tokens.append(s)
            else:
                if w not in seen:
                    seen.add(w); norm_tokens.append(w)
        out = norm_tokens + out

    SINGLE_CHAR_WHITELIST = {
        "爸","妈","儿","女","妻","夫","哥","姐","弟","妹",
        "车","房","钱","猫","狗","书","家","国","城",
        "买","卖","租","吃","喝","玩","学",
        "红","白","黑","蓝","绿",
        "日","月","年","时","今","昨","明",
    }
    all_text = "".join(raw + (raw_norm if text_norm != text else []))
    for ch in all_text:
        if '一' <= ch <= '鿿' and ch in SINGLE_CHAR_WHITELIST:
            if ch not in seen:
                seen.add(ch); out.append(ch)

    return out[:20]


# ============================================================
# 内存 store
# ============================================================
class NovaMemoryStore:
    """Lexical + morphology-based memory, no external deps."""

    def __init__(self) -> None:
        self._chunks: List[str] = []
        self._match_texts: List[str] = []  # content + keywords(可被 LIKE 匹配的全文)
        self._hits: List[int] = []

    def memorize(self, chunks: List[str], keywords: List[str] = None) -> None:
        if keywords is None:
            keywords = [""] * len(chunks)
        for c, kw in zip(chunks, keywords):
            self._chunks.append(c)
            self._match_texts.append((c or "") + " " + (kw or ""))
            self._hits.append(0)

    def recall(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        tokens = tokenize(query)
        if not tokens:
            return self._recent(k)
        scored: List[Tuple[int, int]] = []
        for i, mt in enumerate(self._match_texts):
            cnt = sum(1 for t in tokens if t and t in mt)
            if cnt > 0:
                scored.append((i, cnt))
        if not scored:
            return self._recent(k)
        scored.sort(key=lambda x: (-x[1], x[0]))
        for i, _ in scored:
            self._hits[i] += 1
        return [(self._chunks[i], float(c)) for i, c in scored[:k]]

    def _recent(self, k: int) -> List[Tuple[str, float]]:
        n = min(k, len(self._chunks))
        # 原版:ORDER BY hits DESC, id DESC — 我们用 hits DESC 然后按添加顺序
        if n == 0:
            return []
        idxs = sorted(range(len(self._chunks)), key=lambda i: (-self._hits[i], -i))[:n]
        return [(self._chunks[i], 0.0) for i in idxs]

    def clear(self) -> None:
        self._chunks.clear()
        self._match_texts.clear()
        self._hits.clear()

    def __len__(self) -> int:
        return len(self._chunks)