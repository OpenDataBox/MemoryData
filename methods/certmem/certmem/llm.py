"""Single vLLM engine used for generation (writer/probe gen/answering) and scoring.

Scoring = mean log-prob of the answer tokens given the prompt (teacher forcing) via
vLLM prompt_logprobs. One forward pass per (context, question, answer).
Qwen3 thinking is disabled everywhere (enable_thinking=False).
"""
import math
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


class Engine:
    def __init__(self, cfg):
        self.tok = AutoTokenizer.from_pretrained(cfg.model)
        self.llm = LLM(model=cfg.model,
                       max_model_len=cfg.max_model_len,
                       gpu_memory_utilization=cfg.gpu_memory_utilization,
                       dtype="bfloat16",
                       seed=cfg.seed)
        self.gen_params = SamplingParams(temperature=0.0, max_tokens=256)
        self.score_params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)

    def _chat(self, system, user, assistant_prefix=None, think=False):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self.tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True,
                                            enable_thinking=think)
        if assistant_prefix is not None:
            text += assistant_prefix
        return text

    # ---------- generation ----------
    def generate(self, system, users, max_tokens=256, think=False):
        """think=True enables Qwen3 reasoning; the <think>...</think> block is stripped from the returned text
        and its token count is recorded in self.last_think_tokens (list) for cost accounting."""
        prompts = [self._chat(system, u, think=think) for u in users]
        params = SamplingParams(temperature=0.0, max_tokens=(max_tokens if not think else max(max_tokens, 768)))
        outs = self.llm.generate(prompts, params, use_tqdm=False)
        texts, think_toks = [], []
        for o in outs:
            t = o.outputs[0].text
            if think and "<think>" in t:
                head, _, tail = t.partition("</think>")
                think_toks.append(len(o.outputs[0].token_ids) - len(self.tok(tail, add_special_tokens=False).input_ids))
                t = tail
            else:
                think_toks.append(0)
            texts.append(t.strip())
        self.last_think_tokens = think_toks
        return texts

    # ---------- scoring ----------
    def answer_logprob(self, system, users, answers):
        """Return mean token log-prob of each answer conditioned on its prompt.
        Higher = the reader finds the answer more plausible given the context."""
        prompts, spans = [], []
        for u, a in zip(users, answers):
            prefix = self._chat(system, u)
            n_prefix = len(self.tok(prefix, add_special_tokens=False).input_ids)
            full = prefix + a
            n_full = len(self.tok(full, add_special_tokens=False).input_ids)
            prompts.append(full)
            spans.append((n_prefix, n_full))
        outs = self.llm.generate(prompts, self.score_params, use_tqdm=False)
        res = []
        for o, (s, e) in zip(outs, spans):
            lps = o.prompt_logprobs  # list[None | dict{token_id: Logprob}]
            vals = []
            for pos in range(s, e):
                d = lps[pos]
                if d is None:
                    continue
                # dict has one entry: the actual token
                vals.append(list(d.values())[0].logprob)
            res.append(sum(vals) / max(1, len(vals)) if vals else -math.inf)
        return res


def loss_from_logprob(lp, floor=-6.0):
    """Map mean log-prob to a [0,1] loss. lp=0 -> 0, lp<=floor -> 1."""
    if lp == -math.inf:
        return 1.0
    return min(1.0, max(0.0, -lp / -floor))
