# gh-ai-runner

[![PyPI version](https://badge.fury.io/py/gh-ai-runner.svg)](https://badge.fury.io/py/gh-ai-runner)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**Serverless AI inference via GitHub Actions. No server. No GPU. No infrastructure.**

Run open-source LLMs directly through GitHub's free CI/CD runners — just a GitHub token and a prompt. `gh-ai-runner` handles everything else: repo creation, workflow setup, model downloading, caching, and output retrieval.

Built by [Tanish Chauhan](https://github.com/TanishC4444)

---

## How it works

When you call `ai_call()`:

1. Creates a private GitHub repo (once, automatically)
2. Commits a workflow + inference script into it
3. Dispatches a `workflow_dispatch` GitHub Actions run
4. The runner downloads and caches the model (GGUF quantized, ~0.6 GB)
5. Runs inference via `llama-cpp-python`
6. Uploads the output as an artifact
7. Downloads and returns the output to you as a string

No server is ever running between calls. Each call spins up a fresh GitHub Actions runner, runs inference, and shuts down.

---

## Install
```bash
pip install gh-ai-runner==0.1.3
```

**Requirements:**
- Python 3.9+
- A GitHub account with a personal access token (PAT)
- Token scopes needed: `repo`, `workflow`

---

## Quickstart
```python
from gh_ai_runner import ai_call

result = ai_call(
    github_token="ghp_...",
    prompt="explain recursion in simple terms",
)

print(result)
```

That's it. On first run, `gh-ai-runner` will:
- Create a repo called `ai-inference-runner` on your GitHub account
- Set up the workflow automatically
- Download and cache TinyLlama 1.1B (~0.6 GB)

Subsequent calls reuse the cached repo and model.

---

## Examples

**Basic question**
```python
from gh_ai_runner import ai_call

result = ai_call(
    github_token="ghp_...",
    prompt="what is the difference between a list and a tuple in Python?",
)
print(result)
```

**Custom system prompt**
```python
result = ai_call(
    github_token="ghp_...",
    prompt="explain black holes",
    system="You are a physics professor. Be precise and use analogies.",
    model="llama",
    max_tokens=1024,
)
print(result)
```

**Deterministic output (temperature=0)**
```python
result = ai_call(
    github_token="ghp_...",
    prompt="what is 144 divided by 12?",
    temperature=0.0,
    max_tokens=16,
)
print(result)  # always returns the same answer
```

**Silent mode (no logs)**
```python
result = ai_call(
    github_token="ghp_...",
    prompt="summarize the theory of evolution",
    verbose=False,
)
print(result)
```

**Large output**
```python
result = ai_call(
    github_token="ghp_...",
    prompt="write a detailed essay on the causes of World War I",
    max_tokens=2048,
    temperature=0.5,
    model="llama",
)
print(result)
```

**Multiple calls in sequence**
```python
from gh_ai_runner import ai_call

TOKEN = "ghp_..."

questions = [
    "what is a neural network?",
    "what is gradient descent?",
    "what is backpropagation?",
]

for q in questions:
    answer = ai_call(github_token=TOKEN, prompt=q, verbose=False)
    print(f"Q: {q}\nA: {answer}\n")
```

**Parallel calls (use separate repo per call)**
```python
import threading
from gh_ai_runner import ai_call

TOKEN = "ghp_..."
results = {}

def run(key, prompt, repo):
    results[key] = ai_call(
        github_token=TOKEN,
        prompt=prompt,
        repo_name=repo,
        verbose=False,
    )

t1 = threading.Thread(target=run, args=("q1", "explain DNA", "runner-repo-1"))
t2 = threading.Thread(target=run, args=("q2", "explain RNA", "runner-repo-2"))

t1.start(); t2.start()
t1.join();  t2.join()

print(results["q1"])
print(results["q2"])
```

> **Note:** Parallel calls must use different `repo_name` values. Each repo has its own independent run queue, so calls never interfere with each other.

---

## Parameters

| Parameter | Type | Default | Required | Description |
|---|---|---|---|---|
| `github_token` | `str` | — | Yes | GitHub PAT with `repo` and `workflow` scopes |
| `prompt` | `str` | — | Yes | The message or question to send to the model |
| `model` | `str` | `"tinyllama"` | No | Which model to use. See Models section below |
| `system` | `str` | `"You are a helpful assistant."` | No | System prompt that controls model behavior |
| `max_tokens` | `int` | `512` | No | Max tokens to generate. Hard limit: 4096 |
| `temperature` | `float` | `0.7` | No | Randomness. `0.0` = deterministic, `2.0` = very creative |
| `cache` | `bool` | `True` | No | Cache model weights between runs. Strongly recommended |
| `n_ctx` | `int` | `None` | No | Context window size. Defaults to model built-in. Max: 8192 |
| `repo_name` | `str` | `"ai-inference-runner"` | No | GitHub repo to create or reuse for running inference |
| `verbose` | `bool` | `True` | No | Print step-by-step logs. Set `False` for silent mode |

---

## Models

| Key | Model | Size | Default Context | Max Safe Context |
|---|---|---|---|---|
| `tinyllama` | TinyLlama 1.1B Chat Q4_K_M | 0.6 GB | 2048 tokens | 8192 tokens |
| `llama` | Llama 3.2 1B Instruct Q4_K_M | 0.7 GB | 4096 tokens | 8192 tokens |

Both models are quantized GGUF files hosted publicly on HuggingFace. No HuggingFace token required.

**Which model should I use?**
- `tinyllama` — faster, smaller, good for short factual answers and simple tasks
- `llama` — better instruction-following, better for longer structured outputs and reasoning

---

## Context and output size

The context window (`n_ctx`) is the total number of tokens the model can see at once — this includes your system prompt, your prompt, and the generated output combined.