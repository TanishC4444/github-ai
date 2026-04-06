# github-ai

[![PyPI version](https://badge.fury.io/py/github-ai.svg)](https://badge.fury.io/py/github-ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Serverless AI inference via GitHub Actions. No server, no GPU, no API keys needed.

Built by [Tanish Chauhan](https://github.com/TanishC4444)

## Install

pip install github-ai

## Usage

from github_ai import ai_call

result = ai_call(
    github_token="ghp_...",
    prompt="explain recursion",
)
print(result)

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| github_token | str | required | GitHub PAT with repo + workflow scopes |
| prompt | str | required | Your question or instruction |
| model | str | "tinyllama" | "tinyllama" or "llama" |
| system | str | "You are a helpful assistant." | System prompt |
| max_tokens | int | 512 | Max tokens to generate (hard limit: 4096) |
| temperature | float | 0.7 | 0.0 = deterministic, 2.0 = creative |
| cache | bool | True | Cache model weights between runs |
| n_ctx | int | None | Context window (default: 2048/4096, max: 8192) |
| repo_name | str | "ai-inference-runner" | GitHub repo name to use |
| verbose | bool | True | Print logs. Set False for silent mode |

## Models

| Key | Model | Size | Context |
|---|---|---|---|
| tinyllama | TinyLlama 1.1B Chat Q4_K_M | 0.6 GB | 2048 |
| llama | Llama 3.2 1B Instruct Q4_K_M | 0.7 GB | 4096 |

## Notes

- First run ~5 min (compiles llama-cpp-python + downloads model)
- Cached runs ~1-2 min
- GitHub Actions free tier: unlimited minutes on public repos
- Runner RAM: 7 GB total

## License

MIT