UPDATE_MEMORY_TEMPLATE = """You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. Please read the provided section carefully and update the memory with the new information that helps to answer the problem. Be sure to retain all relevant details from the previous memory while adding any new, useful information.

<problem>
{prompt}
</problem>

<memory>
{memory}
</memory>

<section>
{chunk}
</section>

Updated memory:
"""


FINAL_ANSWER_TEMPLATE = """You are presented with a problem and a previous memory. Please answer the problem based on the previous memory.

If the problem is multiple-choice, respond in the format "Therefore, the answer is X" where X is exactly one uppercase letter from A, B, C, or D.
Otherwise, respond in the format "Therefore, the answer is (insert answer here)".

<problem>
{prompt}
</problem>

<memory>
{memory}
</memory>

Your answer:
"""


NO_MEMORY = "No previous memory"
