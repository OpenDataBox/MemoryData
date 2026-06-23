# System message used across all templates
SYSTEM_MESSAGE = "You are a helpful assistant that can read the context and memorize it for future retrieval."

FACT_EXTRACTION_SHARED_SUFFIX = (
    "Return JSON only in this format: {\"facts\": [\"...\"]}.\n"
    "Rules:\n"
    "- Extract atomic, self-contained facts from the benchmark content.\n"
    "- Ignore system messages and instruction boilerplate; only use benchmark information carried in the user and assistant messages.\n"
    "- Preserve important names, dates, quantities, event order, labels, and causal details.\n"
    "- If the chunk contains relevant information, prefer multiple concise facts rather than an empty list.\n"
    "- Return an empty list only when the chunk is truly non-informative.\n"
    "- Do not include explanations, markdown, or extra keys.\n"
    "- Keep the output language aligned with the input when possible."
)

FACT_EXTRACTION_TEMPLATES = {
    "ruler_qa": (
        "You extract factual memories from benchmark context chunks for document question answering. "
        "Focus on document facts, entity attributes, definitions, and details that could support later QA.\n\n"
        + FACT_EXTRACTION_SHARED_SUFFIX
    ),
    "longbench_mcq": (
        "You extract factual memories from long-context benchmark documents for later multiple-choice question answering. "
        "Focus on document facts, entity attributes, definitions, code or table details, timelines, causal links, and evidence that would help choose between answer options.\n\n"
        + FACT_EXTRACTION_SHARED_SUFFIX
    ),
    "membench_mcq": (
        "You extract factual memories from benchmark dialogue history for later multiple-choice memory questions. "
        "Focus on user facts, assistant recommendations, temporal updates, preferences, entities, names, dates, places, and changes that would help choose between answer options.\n\n"
        + FACT_EXTRACTION_SHARED_SUFFIX
    ),
    "longmemeval": (
        "You extract factual memories from benchmark dialogue history. "
        "Focus on user and assistant statements, preferences, commitments, temporal references, and dialogue facts needed for later questions.\n\n"
        + FACT_EXTRACTION_SHARED_SUFFIX
    ),
    "locomo_qa": (
        "You extract factual memories from benchmark dialogue history. "
        "Focus on user and assistant statements, preferences, commitments, temporal references, and dialogue facts needed for later questions.\n\n"
        + FACT_EXTRACTION_SHARED_SUFFIX
    ),
    "eventqa": (
        "You extract factual memories from benchmark book excerpts. "
        "Focus on events, chronology, participants, locations, state changes, and causal details needed to infer what happens next.\n\n"
        + FACT_EXTRACTION_SHARED_SUFFIX
    ),
    "in_context_learning": (
        "You extract factual memories from benchmark in-context learning examples. "
        "Focus on mappings between inputs and labels, label definitions, patterns, and discriminative cues needed to classify later examples.\n\n"
        + FACT_EXTRACTION_SHARED_SUFFIX
    ),
    "recsys_redial": (
        "You extract factual memories from benchmark recommendation dialogues. "
        "Focus on user preferences, dislikes, watched titles, mentioned movies, genres, actors, and recommendation constraints.\n\n"
        + FACT_EXTRACTION_SHARED_SUFFIX
    ),
    "infbench_sum": (
        "You extract factual memories from benchmark book content for later summarization. "
        "Focus on major events, characters, relationships, themes, and outcomes that should be preserved in a summary.\n\n"
        + FACT_EXTRACTION_SHARED_SUFFIX
    ),
    "detective_qa": (
        "You extract factual memories from benchmark detective stories. "
        "Focus on clues, suspects, evidence, alibis, motives, timelines, and relationships relevant to solving later questions.\n\n"
        + FACT_EXTRACTION_SHARED_SUFFIX
    ),
    "factconsolidation": (
        "You extract factual memories from benchmark knowledge-pool entries. "
        "Focus on facts, serial numbers, entities, and conflict-relevant updates. Preserve serial numbers and recency cues verbatim so newer facts can override older ones later.\n\n"
        + FACT_EXTRACTION_SHARED_SUFFIX
    ),
}

MEMORY_ANSWER_TEMPLATES = {
    "ruler_qa": (
        "Use the retrieved memories below to answer the question. Only give me the answer and do not output any other words.\n\n"
        "Retrieved memories:\n{memories}\n\n"
        "Question: {question}\n\nAnswer:"
    ),
    "longbench_mcq": (
        "Use the retrieved memories below to answer the multiple-choice question. Reply with exactly one uppercase letter: A, B, C, or D. Do not explain your answer.\n\n"
        "Retrieved memories:\n{memories}\n\n"
        "{question}\n\nAnswer:"
    ),
    "membench_mcq": (
        "The retrieved memories below are dialogue history between you and a user. "
        "Use them to answer the multiple-choice memory question. Reply with exactly one uppercase letter: A, B, C, or D.\n\n"
        "Retrieved memories:\n{memories}\n\n"
        "{question}\n\n"
        "Answer:"
    ),
    "longmemeval": (
        "The retrieved memories below are chat history between you and a user. Based on that history, answer the question as concisely as you can, using a single phrase if possible.\n\n"
        "Retrieved memories:\n{memories}\n\n"
        "{question}\n\nAnswer:"
    ),
    "locomo_qa": (
        "The retrieved memories below are chat history between you and a user. Based on that history, answer the question as concisely as you can, using a single phrase if possible.\n\n"
        "Retrieved memories:\n{memories}\n\n"
        "{question}\n\nAnswer:"
    ),
    "eventqa": (
        "Based on the retrieved memories below, complete the task.\n\n"
        "Retrieved memories:\n{memories}\n\n"
        "{question}\n\nThe event that happens next is:"
    ),
    "in_context_learning": (
        "Use the provided mapping in the retrieved memories to assign a numerical label to the input. Only output \"label: {{label}}\" and nothing else.\n\n"
        "Retrieved memories:\n{memories}\n\n"
        "{question}\n\nlabel:"
    ),
    "recsys_redial": (
        "Pretend you are a movie recommender system. Use the retrieved memories below to recommend movies based on the conversation. Reply with 20 recommendations and no extra sentences.\n\n"
        "Retrieved memories:\n{memories}\n\n"
        "Here is the conversation: {question}\n\nThe recommendations are:\n"
    ),
    "infbench_sum": (
        "You are given retrieved memories from a book. Use them to complete the task below.\n\n"
        "Retrieved memories:\n{memories}\n\n"
        "{question}\n\nNow summarize the book."
    ),
    "detective_qa": (
        "Based on the retrieved memories below, answer the question and follow the required output format exactly.\n\n"
        "Retrieved memories:\n{memories}\n\n"
        "{question}\n\n"
    ),
    "factconsolidation": (
        "Pretend you are a knowledge management system. Each fact in the retrieved memories is provided with a serial number, and newer facts have larger serial numbers. Resolve conflicts by using the newest relevant fact only. Give a concise answer using only the retrieved memories rather than real-world knowledge.\n\n"
        "Retrieved memories:\n{memories}\n\n"
        "Question: Based on the provided Knowledge Pool, {question}\nAnswer:"
    ),
}

# Base templates with placeholders for agent-specific variations
BASE_TEMPLATES = {
    "ruler_qa": {
        "system": SYSTEM_MESSAGE,
        "memorize": "Dialogue between User and Assistant {time_stamp}\n<User> The following context is the documents I have read: \n{context}\n <Assistant> I have learned the documents and I will answer the question you ask.",
        "query": {
            "long_context_agent": "Answer the question based on the memorized documents. Only give me the answer and do not output any other words. \n\nQuestion: {question} \n\n Answer:",
            "rag_agent": "Answer the question based on the memorized documents. Only give me the answer and do not output any other words. \n\n Now Answer the Question: {question}",
            "agentic_memory_agent": "Search Archival Memory and answer my question. Only give me the answer and do not output any other words. \n\nQuestion: {question} \n\n Answer:",
            "memos_agent": "Based on your stored memories, answer my question. Only give me the answer and do not output any other words. \n\nQuestion: {question} \n\n Answer:",
            "memagent": "Based on the memorized context, answer my question. Only give me the answer and do not output any other words. \n\nQuestion: {question} \n\n Answer:",
        },
    },
    "longbench_mcq": {
        "system": SYSTEM_MESSAGE,
        "memorize": "Dialogue between User and Assistant {time_stamp}\n<User> The following context is the document I have read: \n{context}\n <Assistant> I have learned the document and I will answer the question you ask.",
        "query": {
            "long_context_agent": "Answer the following multiple-choice question based on the memorized document. Reply with exactly one uppercase letter: A, B, C, or D. Do not explain your answer.\n\nQuestion: {question}\nA. {choice_A}\nB. {choice_B}\nC. {choice_C}\nD. {choice_D}\n\nAnswer:",
            "rag_agent": "Answer the following multiple-choice question based on the memorized document. Reply with exactly one uppercase letter: A, B, C, or D. Do not explain your answer.\n\nQuestion: {question}\nA. {choice_A}\nB. {choice_B}\nC. {choice_C}\nD. {choice_D}\n\nAnswer:",
            "agentic_memory_agent": "Search Archival Memory and answer the following multiple-choice question. Reply with exactly one uppercase letter: A, B, C, or D. Do not explain your answer.\n\nQuestion: {question}\nA. {choice_A}\nB. {choice_B}\nC. {choice_C}\nD. {choice_D}\n\nAnswer:",
            "memos_agent": "Based on your stored memories, answer the following multiple-choice question. Reply with exactly one uppercase letter: A, B, C, or D. Do not explain your answer.\n\nQuestion: {question}\nA. {choice_A}\nB. {choice_B}\nC. {choice_C}\nD. {choice_D}\n\nAnswer:",
            "memagent": "Based on the memorized context, answer the following multiple-choice question. Reply with exactly one uppercase letter: A, B, C, or D. Do not explain your answer.\n\nQuestion: {question}\nA. {choice_A}\nB. {choice_B}\nC. {choice_C}\nD. {choice_D}\n\nAnswer:",
        },
    },
    "membench_mcq": {
        "system": SYSTEM_MESSAGE,
        "memorize": "Dialogue between User and Assistant \n<User> The following context is the conversation between the user and the assistant: \n{context}\n <Assistant> I have memorized the conversation and I will answer the question you ask.",
        "query": {
            "long_context_agent": "Answer the following multiple-choice memory question based on the memorized conversation. Reply with exactly one uppercase letter: A, B, C, or D.\n\nQuestion: (current time is {question_dates}) {question}\nA. {choice_A}\nB. {choice_B}\nC. {choice_C}\nD. {choice_D}\n\nAnswer:",
            "rag_agent": "Answer the following multiple-choice memory question based on the memorized conversation. Reply with exactly one uppercase letter: A, B, C, or D.\n\nQuestion: (current time is {question_dates}) {question}\nA. {choice_A}\nB. {choice_B}\nC. {choice_C}\nD. {choice_D}\n\nAnswer:",
            "agentic_memory_agent": "Search Archival Memory and answer the following multiple-choice memory question. Reply with exactly one uppercase letter: A, B, C, or D.\n\nQuestion: (current time is {question_dates}) {question}\nA. {choice_A}\nB. {choice_B}\nC. {choice_C}\nD. {choice_D}\n\nAnswer:",
            "memos_agent": "Based on your stored memories of the conversation history, answer the following multiple-choice memory question. Reply with exactly one uppercase letter: A, B, C, or D.\n\nQuestion: (current time is {question_dates}) {question}\nA. {choice_A}\nB. {choice_B}\nC. {choice_C}\nD. {choice_D}\n\nAnswer:",
            "memagent": "Based on the memorized conversation history, answer the following multiple-choice memory question. Reply with exactly one uppercase letter: A, B, C, or D.\n\nQuestion: (current time is {question_dates}) {question}\nA. {choice_A}\nB. {choice_B}\nC. {choice_C}\nD. {choice_D}\n\nAnswer:",
        },
    },
    "longmemeval": {
        "system": SYSTEM_MESSAGE,
        "memorize": "Dialogue between User and Assistant \n<User> The following context is the conversation between the user and the assistant: \n{context}\n <Assistant> I have memorized the conversation and I will answer the question you ask.",
        "query": {
            "long_context_agent": "The history chats are between you and a user. Based on the relevant chat history, answer the question as concisely as you can, using a single phrase if possible.\n\n {question} \n\n Answer:",
            "rag_agent": "The history chats are between you and a user. Based on the relevant chat history, answer the question as concisely as you can, using a single phrase if possible.\n\n {question} \n\n Answer:",
            "agentic_memory_agent": "Search Archival Memory and answer the question as concisely as you can, using a single phrase if possible.\n\n {question} \n\n Answer:",
            "memos_agent": "Based on your stored memories of the conversation history, answer the question as concisely as you can, using a single phrase if possible.\n\n {question} \n\n Answer:",
            "memagent": "Based on the memorized conversation history, answer the question as concisely as you can, using a single phrase if possible.\n\n {question} \n\n Answer:",
        },
    },
    "locomo_qa": {
        "system": SYSTEM_MESSAGE,
        "memorize": "Dialogue between User and Assistant \n<User> The following context is the conversation between the user and the assistant: \n{context}\n <Assistant> I have memorized the conversation and I will answer the question you ask.",
        "query": {
            "long_context_agent": "The history chats are between you and a user. Based on the relevant chat history, answer the question as concisely as you can, using a single phrase if possible.\n\n {question} \n\n Answer:",
            "rag_agent": "The history chats are between you and a user. Based on the relevant chat history, answer the question as concisely as you can, using a single phrase if possible.\n\n {question} \n\n Answer:",
            "agentic_memory_agent": "Search Archival Memory and answer the question as concisely as you can, using a single phrase if possible.\n\n {question} \n\n Answer:",
            "memos_agent": "Based on your stored memories of the conversation history, answer the question as concisely as you can, using a single phrase if possible.\n\n {question} \n\n Answer:",
            "memagent": "Based on the memorized conversation history, answer the question as concisely as you can, using a single phrase if possible.\n\n {question} \n\n Answer:",
        },
    },
    "eventqa": {
        "system": SYSTEM_MESSAGE,
        "memorize": "Dialogue between User and Assistant {time_stamp}\n<User> The following context is the book excerpt: \n{context}\n <Assistant> I have read the book excerpt and I will answer the question you ask.",
        "query": {
            "long_context_agent": "Based on the context you memorized, complete the task below:\n\n{question}\n\n The event that happens next is:",
            "rag_agent": "Based on the context you memorized, complete the task below:\n\n{question}\n\n The event that happens next is:",
            "agentic_memory_agent": "Search Archival Memory, complete the task below:\n\n{question}\n\n The event that happens next is:",
            "memos_agent": "Based on your stored memories, complete the task below:\n\n{question}\n\n The event that happens next is:",
            "memagent": "Based on the memorized context, complete the task below:\n\n{question}\n\n The event that happens next is:",
        },
    },
    "in_context_learning": {
        "system": SYSTEM_MESSAGE,
        "memorize": "Dialogue between User and Assistant {time_stamp} \n<User> The following context is the examples I have learned: \n{context}\n <Assistant> I have learned the examples and I will answer the question you ask.",
        "query": {
            "long_context_agent": "Use the provided mapping from the context to numerical label to assign a numerical label to the context. Only output \"label: {{label}}\" and nothing else. \n\n{question} \n\n label:",
            "rag_agent": "Use the provided mapping from the context to numerical label to assign a numerical label to the context. Only output \"label: {{label}}\" and nothing else. \n\nQuestion:{question} \n\n label:",
            "agentic_memory_agent": "Search Archival Memory and use the provided mapping from the context to numerical label to assign a numerical label to the context. Only output \"label: {{label}}\" and nothing else. \n\n{question} \n\n label:",
            "memos_agent": "Based on your stored memories, use the provided mapping from the context to numerical label to assign a numerical label to the context. Only output \"label: {{label}}\" and nothing else. \n\n{question} \n\n label:",
            "memagent": "Based on the memorized context, use the provided mapping from the context to numerical label to assign a numerical label to the context. Only output \"label: {{label}}\" and nothing else. \n\n{question} \n\n label:",
        },
    },
    "recsys_redial": {
        "system": SYSTEM_MESSAGE,
        "memorize": "Dialogue between User and Assistant {time_stamp} \n<User> The following context is the dialogues between a user and recommender system: \n{context}\n <Assistant> I have memorized the dialogues and I will answer the question you ask.",
        "query": {
            "long_context_agent": "Pretend you are a movie recommender system. You need to recommend movies based on the dialogues you have memorized. Now I will give you a new conversation between a user and you (a recommender system). Based on the conversation, you reply me with 20 recommendations without extra sentences. \n\nFor Example:\n\n[Conversation]\n\nThe recommendations are: \n1.movie1\n2.movie2\n...\n\n Here is the conversation: {question} \n\n The recommendations are: \n",
            "rag_agent": "Pretend you are a movie recommender system. You need to recommend movies based on the dialogues you have memorized. Now I will give you a new conversation between a user and you (a recommender system). Based on the conversation, you reply me with 20 recommendations without extra sentences. \n\nFor Example:\n\n[Conversation]\n\nThe recommendations are: \n1.movie1\n2.movie2\n...\n\n Here is the conversation: {question} \n\n The recommendations are: \n",
            "agentic_memory_agent": "Pretend you are a movie recommender system. You need to recommend movies based on the dialogues you have memorized. Now I will give you a new conversation between a user and you (a recommender system). Search Archival Memory, you reply me with 20 recommendations without extra sentences. \n\nFor Example:\n\n[Conversation]\n\nThe recommendations are: \n1.movie1\n2.movie2\n...\n\n Here is the conversation: {question} \n\n The recommendations are: \n",
            "memos_agent": "Pretend you are a movie recommender system. You need to recommend movies based on the dialogues you have memorized. Now I will give you a new conversation between a user and you (a recommender system). Based on your stored memories, you reply me with 20 recommendations without extra sentences. \n\nFor Example:\n\n[Conversation]\n\nThe recommendations are: \n1.movie1\n2.movie2\n...\n\n Here is the conversation: {question} \n\n The recommendations are: \n",
            "memagent": "Pretend you are a movie recommender system. You need to recommend movies based on the dialogues you have memorized. Now I will give you a new conversation between a user and you (a recommender system). Based on the memorized context, you reply me with 20 recommendations without extra sentences. \n\nFor Example:\n\n[Conversation]\n\nThe recommendations are: \n1.movie1\n2.movie2\n...\n\n Here is the conversation: {question} \n\n The recommendations are: \n",
        },
    },
    "infbench_sum": {
        "system": SYSTEM_MESSAGE,
        "memorize": "Dialogue between User and Assistant {time_stamp} \n<User> The following context is the book I have read: \n{context}\n <Assistant> I have read the book and I will answer the question you ask.",
        "query": {
            "long_context_agent": "You are given a book above and you are tasked to summarize it. \n\n{question} \n\n Now summarize the book.",
            "rag_agent": "You are given a book above and you are tasked to summarize it. \n\n{question} \n\n Now summarize the book.",
            "agentic_memory_agent": "You are given a book above and you are tasked to summarize it. \n\n{question} \n\n Now summarize the book.",
            "memos_agent": "Based on your stored memories of the book, summarize it. \n\n{question} \n\n Now summarize the book.",
            "memagent": "Based on the memorized book content, summarize it. \n\n{question} \n\n Now summarize the book.",
        },
    },
    "detective_qa": {
        "system": SYSTEM_MESSAGE,
        "memorize": "Dialogue between User and Assistant {time_stamp} \n<User> The following context is the book I have read: \n{context}\n <Assistant> I have read the book and I will answer the question you ask.",
        "query": {
            "long_context_agent": "Based on the context you memorized, answer the question below. You are required to answer the question based on the strict output format.\n\n {question} \n\n",
            "rag_agent": "Based on the context you memorized, answer the question below. You are required to answer the question based on the strict output format.\n\n {question} \n\n",
            "agentic_memory_agent": "Search Archival Memory and answer the question below. You are required to answer the question based on the strict output format.\n\n {question} \n\n",
            "memos_agent": "Based on your stored memories, answer the question below. You are required to answer the question based on the strict output format.\n\n {question} \n\n",
            "memagent": "Based on the memorized context, answer the question below. You are required to answer the question based on the strict output format.\n\n {question} \n\n",
        },
    },
    "factconsolidation": {
        "system": SYSTEM_MESSAGE,
        "memorize": "Dialogue between User and Assistant {time_stamp} \n<User> The following context is the facts I have learned: \n{context}\n <Assistant> I have learned the facts and I will answer the question you ask.",
        "query": {
            "long_context_agent": "Pretend you are a knowledge management system. Each fact in the knowledge pool is provided with a serial number at the beginning, and the newer fact has larger serial number. \n You need to solve the conflicts of facts in the knowledge pool by finding the newest fact with larger serial number. You need to answer a question based on this rule. You should give a very concise answer without saying other words for the question **only** from the knowledge pool you have memorized rather than the real facts in real world. \n\nFor example:\n\n [Knowledge Pool] \n\n Question: Based on the provided Knowledge Pool, what is the name of the current president of Russia? \nAnswer: Donald Trump \n\n Now Answer the Question: Based on the provided Knowledge Pool, {question} \nAnswer:",
            "rag_agent": "Pretend you are a knowledge management system. Each fact in the knowledge pool is provided with a serial number at the beginning, and the newer fact has larger serial number. \n You need to solve the conflicts of facts in the knowledge pool by finding the newest fact with larger serial number. You need to answer a question based on this rule. You should give a very concise answer without saying other words for the question **only** from the knowledge pool you have memorized rather than the real facts in real world. \n\nFor example:\n\n [Knowledge Pool] \n\n Question: Based on the provided Knowledge Pool, what is the name of the current president of Russia? \nAnswer: Donald Trump \n\n Now Answer the Question: Based on the provided Knowledge Pool, {question} \nAnswer:",
            "agentic_memory_agent": "Pretend you are a knowledge management system. Each fact in the  Archival Memory is provided with a serial number at the beginning, and the newer fact has larger serial number. \n You need to solve the conflicts of facts in the Archival Memory by finding the newest fact with larger serial number. You need to answer a question based on this rule. You should give a very concise answer without saying other words for the question **only** from the knowledge pool you have memorized rather than the real facts in real world. \n\nFor example:\n\n [Archival Memory] \n\n Question: Based on the Archival Memory, what is the name of the current president of Russia? \nAnswer: Donald Trump \n\n Now Answer the Question: Based on the  Archival Memory, {question} \nAnswer:",
            "memos_agent": "Pretend you are a knowledge management system. Each fact in your stored memories is provided with a serial number at the beginning, and the newer fact has larger serial number. \n You need to solve the conflicts of facts in your stored memories by finding the newest fact with larger serial number. You need to answer a question based on this rule. You should give a very concise answer without saying other words for the question **only** from the knowledge pool you have memorized rather than the real facts in real world. \n\nFor example:\n\n [Knowledge Pool] \n\n Question: Based on the provided Knowledge Pool, what is the name of the current president of Russia? \nAnswer: Donald Trump \n\n Now Answer the Question: Based on the provided Knowledge Pool, {question} \nAnswer:",
            "memagent": "Pretend you are a knowledge management system. Each fact in the memorized context is provided with a serial number at the beginning, and the newer fact has larger serial number. \n You need to solve the conflicts of facts in the memorized context by finding the newest fact with larger serial number. You need to answer a question based on this rule. You should give a very concise answer without saying other words for the question **only** from the knowledge pool you have memorized rather than the real facts in real world. \n\nFor example:\n\n [Knowledge Pool] \n\n Question: Based on the provided Knowledge Pool, what is the name of the current president of Russia? \nAnswer: Donald Trump \n\n Now Answer the Question: Based on the provided Knowledge Pool, {question} \nAnswer:",
        },
    },
}

# Order matters: more specific prefixes must come before "rag", because agent names like
# "Agentic_memory_self_rag" and "Structure_rag_mem0" contain the substring "rag".
AGENT_TYPE_MAPPING = [
    ("Long_context_agent", "long_context_agent"),
    ("Agentic_memory", "agentic_memory_agent"),
    ("letta", "agentic_memory_agent"),
    ("mem0", "agentic_memory_agent"),
    ("cognee", "agentic_memory_agent"),
    ("memtree", "agentic_memory_agent"),
    ("memochat", "agentic_memory_agent"),
    ("zep", "agentic_memory_agent"),
    ("simplemem", "agentic_memory_agent"),
    ("lightmem", "agentic_memory_agent"),
    ("memagent", "agentic_memory_agent"),
    ("a_mem", "agentic_memory_agent"),
    ("memtree", "agentic_memory_agent"),
    ("MemOS", "memos_agent"),
    ("memagent", "memagent"),
    ("rag", "rag_agent"),
]

DATASET_MAPPING = {
    ("ruler_", "qa"): "ruler_qa",
    ("longbench",): "longbench_mcq",
    ("membench_",): "membench_mcq",
    ("icl_",): "in_context_learning",
    ("infbench_", "sum"): "infbench_sum",
    ("eventqa_",): "eventqa",
    ("recsys_", "redial"): "recsys_redial",
    ("longmemeval_",): "longmemeval",
    ("locomo_",): "locomo_qa",
    ("factconsolidation_",): "factconsolidation",
    ("detective_", "qa"): "detective_qa",
}


def normalize_agent_name(agent_name):
    """Normalize agent name to standard form."""
    normalized_input = agent_name.lower()
    for pattern, normalized_name in AGENT_TYPE_MAPPING:
        if pattern.lower() in normalized_input:
            return normalized_name
    raise NotImplementedError(f"Unknown agent type: {agent_name}")


def normalize_dataset_name(sub_dataset):
    """Normalize dataset name to standard form."""
    normalized_input = sub_dataset.lower()
    for patterns, normalized_name in DATASET_MAPPING.items():
        if all(pattern in normalized_input for pattern in patterns):
            return normalized_name
    raise NotImplementedError(f"Unknown dataset: {sub_dataset}")


def get_template(sub_dataset, template_name, agent_name):
    """
    Get template for specified agent, dataset, and template type.

    Args:
        sub_dataset: Dataset identifier
        template_name: Type of template ('system', 'memorize', 'query', 'fact_extraction', 'memory_answer')
        agent_name: Agent type identifier

    Returns:
        Template string
    """
    normalized_dataset = normalize_dataset_name(sub_dataset)

    if template_name == "fact_extraction":
        return FACT_EXTRACTION_TEMPLATES[normalized_dataset]
    if template_name == "memory_answer":
        return MEMORY_ANSWER_TEMPLATES[normalized_dataset]

    normalized_agent = normalize_agent_name(agent_name)
    base_template = BASE_TEMPLATES[normalized_dataset][template_name]

    if isinstance(base_template, dict):
        return base_template[normalized_agent]
    return base_template
