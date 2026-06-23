AGGREGATE_PROMPT = """
You will receive two pieces of information.
New Information is detailed, and Existing Information is a summary from {n_children} previous entries.
Your task is to merge these into a single, cohesive summary that highlights the most important insights.
Focus on the key points from both inputs.
Ensure the final summary combines the insights from both pieces of information.
If the number of previous entries in Existing Information is accumulating (more than 2), focus on summarizing more concisely, only capturing the overarching theme, and getting more abstract in your summary.
Output the summary directly.

[New Information]
{new_content}
[Existing Information (from {n_children} previous entries)]
{current_content}

IMPORTANT! Don't output additional commentary, explanations, or unrelated information. Provide only the exact information or output requested.
[Output Summary]
"""
