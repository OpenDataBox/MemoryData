# Default settings for llama.cpp-compatible generation endpoints.
SIMPLE = {
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
    # "n_predict": 3072,
}
