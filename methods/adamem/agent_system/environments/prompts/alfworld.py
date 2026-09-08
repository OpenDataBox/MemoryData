# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# --------------------- ALFWorld --------------------- #
ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_TEMPLATE_NO_HIS_WITHOUT_ACTION = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].
"""

ALFWORLD_ACTION_INSTR = """
Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_TEMPLATE_WITHOUT_ACTION = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].
"""

ALFWORLD_TEMPLATE_REFLECTION = """
You need to synthesize step-level strategy before next action generation. You will be given K={k} retrieved experiences. Each item has the past state, past action, remaining trajectory and final outcome ('success' or 'failure').

Goal:
    1) Extract a concise, actionable strategy tailored to current state by learning from retrieved experiences.
    2) Immediately apply that strategy to produce the next action.

Reasoning (must be enclosed in <think>...</think>):
<think>
1) Similarity analysis:
   - Analyze how current state is similar to and different from each past state.
2) Outcome-aware reflection:
   - Explain why successful examples likely worked and why failed ones didn’t.
3) Strategy abstraction:
   - Synthesize 1–3 actionable, non-generic strategy bullets tailored to current state (avoid copying raw text).
4) Utilization:
   - Use the strategy to reason toward one concrete next action that best advances the task now.
</think>

Constraints:
- Be specific but not overfit: avoid instance-specific hacks unless explicitly required by current state.
- If evidence is weak or conflicting, state the uncertainty briefly in the strategy bullets and choose the safest high-utility action.
- Do not repeat long quotes from retrieved experiences; abstract them.

Retrieved Experiences:
{retrieved_exp}
"""

ALFWORLD_TEMPLATE_WITH_OPTIONAL_MEMORY = """

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.

MEMORY RETRIEVAL OPTION:
You have access to a memory system that stores past experiences from similar situations. Using memory can often clarify ambiguity and suggest effective strategies.

After presenting your reasoning and initial proposed action, consider whether retrieving related past experiences could improve your decision. Retrieval may return both successful and failed examples—successful ones illustrate proven approaches, while failed ones highlight common pitfalls to avoid. Both are valuable for informed decision-making.

We recommend retrieving memory in ambiguous or uncertain situations, such as when:

- Multiple possible actions exist without a clear best choice.

- You are unsure about the next step (your reasoning includes words like “maybe,” “uncertain,” or “not sure”).

- The task involves a sequence of actions where past examples could serve as a concrete template.

Your output format should be:
<think>...your reasoning...</think>
<action>...your chosen action...</action>
<memory_request_rationale>Briefly explain whether and why memory would help here (1-2 sentences)</memory_request_rationale>
<request_memory>yes or no</request_memory>
"""

ALFWORLD_TEMPLATE_STRATEGY_GENERATION = """
You need to synthesize contextualized, non-generic strategies from retrieved past experiences. You are given K = {k} retrieved experiences, each containing the past state, past action, remaining trajectory, and final outcome (“success” or “failure”).

Goal: Derive a concise, actionable, and non-generic strategy tailored to the current state by learning from the retrieved experiences. This strategy will later guide the next action.

Constraints:

- Be specific but not overfitted — avoid instance-specific shortcuts unless clearly relevant to the current state.

- If the evidence from retrieved examples is weak or conflicting, briefly note that uncertainty within the strategy.

- Do not quote or copy from retrieved experiences; abstract their key insights.

Retrieved Experiences:
{retrieved_exp}

First, provide your reasoning enclosed in <think>...</think>. Analyze how the current state resembles or differs from each past state, and explain why successful examples likely worked and why failed ones did not.

Then, within a single <strategy> block, list 1–3 concise, actionable, context-specific strategy bullets tailored to the current state. Do not produce multiple <strategy> tags. Do not generate the next action itself.

Output format:

<think>...your reasoning...</think>
<strategy>
- ...
- ...
- ...
</strategy>
"""

ALFWORLD_TEMPLATE_ACTION_FROM_STRATEGY = """
Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.

Below are some strategy items that are accumulated from past interactions from the environment that may be helpful to solve the task. You can use it when you feel it's relevant. In each step, please first explicitly discuss if you want to use each strategy item or not, and then take action.\n\n{strategy}
"""

ALFWORLD_TEMPLATE_STRATEGY_NO_RETRIEVAL = """
You need to synthesize a contextualized strategy based on the current observation.

Goal: Derive a concise, actionable strategy tailored to the current state.

Constraints:
- Be specific but not overfitted — avoid instance-specific shortcuts unless clearly relevant to the current state.
- Provide 1–3 concise, actionable strategy bullets tailored to the current state.

First, provide your reasoning enclosed in <think>...</think>. Analyze the current state and plan a strategy.

Then, within a single <strategy> block, list 1–3 concise, actionable strategy bullets. Do not produce multiple <strategy> tags. Do not generate the next action itself.

Output format:
<think>...your reasoning...</think>
<strategy>
- ...
- ...
- ...
</strategy>
"""

ALFWORLD_TEMPLATE_DIRECT_ACTION_FROM_RETRIEVAL = """
Here are some task solving trajectories on similar tasks:

Retrieved Experiences:
{retrieved_exp}

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_TEMPLATE_STRATEGY_REFRESH_DECISION = """
Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.

Below are some strategy items that are accumulated from past interactions from the environment that may be helpful to solve the task. You can use it when you feel it's relevant. In each step, please first explicitly discuss if you want to use each strategy item or not, and then take action.\n\n{current_strategy}

-----

### STRATEGY REFRESH OPTION

If you believe the current strategy is outdated or misaligned, you may request to synthesize a new strategy using your memory system.

**Implication of Refreshing:**
If you trigger a refresh (`yes`), your currently proposed action will **not** be executed. Instead, the system will update the strategy, and you will be given the opportunity to **make a new decision for this exact step** using the updated context.

**Recommended Criteria for Refreshing:**
  * **Observation Mismatch:** The current state/observation contradicts the expectations set by your existing strategy.
  * **Persistent Failure:** You have attempted similar actions repeatedly without making progress toward the goal.
  * **Critical Ambiguity:** The current situation presents a novel obstacle or edge case where you lack the context to proceed safely.

**Instructions:**

1.  Generate a **proposed action** based on the *current* strategy.
2.  Decide if a refresh is strictly necessary based on the criteria above.

**Output format:**
<think>...reasoning...</think>
<action>...your proposed action...</action>
<refresh_decision>yes or no</refresh_decision>
"""

ALFWORLD_TEMPLATE_UPDATE_DECISION = """\

### ADAPTER UPDATE OPTION

Adapter history so far:
{adapter_history}

After generating your action, you must also decide how to update your learning adapter.

**Implication of Each Decision:**
- If you choose `keep`: Action will be re-generated using the current adapter. No retraining occurs.
- If you choose `update`: Action will be re-generated using a newly retrained adapter on trajectories retrieved from the current state.
- If you choose `reset`: The current adapter will be unloaded, and action will be re-generated using the base model.

**Recommended Criteria:**
  * **keep**: The current adapter is performing well, or the retrieved trajectories are similar to what you have already learned from. Prefer `keep` when you are making consistent progress.
  * **update**: The current situation has diverged significantly from what your adapter was trained on — e.g., you are stuck in a loop, the environment is behaving unexpectedly, or retrieved trajectories suggest a clearly different strategy than what you have been following.
  * **reset**: The adapter is actively hurting performance — e.g., it is consistently biasing you toward wrong actions, or you suspect it has overfit to a different task type than the one you are currently solving.

**Instructions:**

1. Generate a **proposed action** based on your current reasoning.
2. Decide which update decision best fits the current situation based on the criteria above.
3. Include your rationale for the update decision explicitly inside your `<think>` block.

**Output format:**
<think>...reasoning about the action and why you chose keep/update/reset...</think>
<action>...your proposed action...</action>
<update_decision>keep, update, or reset</update_decision>"""

ALFWORLD_TEMPLATE_UPDATE_DECISION_NO_RESET = """\

### ADAPTER UPDATE OPTION

Adapter history so far:
{adapter_history}

After generating your action, you must also decide how to update your learning adapter.

**Implication of Each Decision:**
- If you choose `keep`: Action will be re-generated using the base model. No retraining occurs.
- If you choose `update`: Action will be re-generated using a newly retrained adapter on trajectories retrieved from the current state.

**Recommended Criteria:**
  * **keep**: The base model is performing adequately for the current situation.
  * **update**: You believe retraining on retrieved trajectories would improve performance — e.g., the task involves a clear pattern that past experiences can guide.

**Instructions:**

1. Generate a **proposed action** based on your current reasoning.
2. Decide which update decision best fits the current situation based on the criteria above.
3. Include your rationale for the update decision explicitly inside your `<think>` block.

**Output format:**
<think>...reasoning about the action and why you chose keep or update...</think>
<action>...your proposed action...</action>
<update_decision>keep or update</update_decision>"""

ALFWORLD_TEMPLATE_TRANSIENT_UPDATE = """\

### ON-DEMAND ADAPTATION OPTION

After proposing your action, decide whether to request an on-the-fly adaptation: the system will retrieve similar past trajectories, fine-tune on them, and re-generate your action using the updated model. The adapter is used only for this step and discarded immediately after.

Request adaptation when the current situation is ambiguous or unfamiliar and past examples would likely suggest a better action. Skip it when you are already confident.

**Output format:**
<think>...reasoning about the action and whether past experience would help...</think>
<action>...your proposed action...</action>
<request_update>yes or no</request_update>"""