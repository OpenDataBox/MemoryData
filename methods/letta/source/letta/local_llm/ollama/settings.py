# Default settings for Ollama-compatible generation endpoints.
SIMPLE = {
    "options": {
        "stop": [
            "\nUSER:",
            "\nASSISTANT:",
            "\nFUNCTION RETURN:",
            "\nUSER",
            "\nASSISTANT",
            "\nFUNCTION RETURN",
            "\nFUNCTION",
            "\nFUNC",
            "<|im_start|>",
            "<|im_end|>",
            "<|im_sep|>",
            # '\n' +
            # '</s>',
            # '<|',
            # '\n#',
            # '\n\n\n',
        ],
        # "num_ctx": LLM_MAX_TOKENS,
    },
    "stream": False,
    # turn off Ollama's own prompt formatting
    "system": "",
    "template": "{{ .Prompt }}",
    # "system": None,
    # "template": None,
    "context": None,
}
