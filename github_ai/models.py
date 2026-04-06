MODELS = {
    "tinyllama": {
        "name":      "TinyLlama 1.1B Chat",
        "size_gb":   0.6,
        "n_ctx":     2048,
        "max_n_ctx": 8192,
        "url":       "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "filename":  "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    },
    "llama": {
        "name":      "Llama 3.2 1B Instruct",
        "size_gb":   0.7,
        "n_ctx":     4096,
        "max_n_ctx": 8192,
        "url":       "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "filename":  "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    },
}

RUNNER_RAM_GB     = 7
RAM_PER_1K_CTX_GB = 0.2
MAX_TOKENS_LIMIT  = 4096
MAX_TEMPERATURE   = 2.0
MIN_TEMPERATURE   = 0.0