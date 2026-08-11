# gh-ai-runner

[![PyPI](https://img.shields.io/pypi/v/gh-ai-runner)](https://pypi.org/project/gh-ai-runner/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Python package for serverless-style local-LLM inference through GitHub Actions.

## Overview

`gh-ai-runner` creates or reuses a GitHub repository, provisions an inference workflow, runs a quantized open-source model on a GitHub Actions runner, and returns the generated output. It is designed for workloads that do not need a continuously running inference server.

## Features

- GitHub Actions-based inference
- Automatic runner repository/workflow setup
- GGUF model support
- Model caching between runs
- Multiple model configurations
- Configurable system prompts, context, temperature, and output length
- Python API returning generated text

## Prerequisites

- Python 3.9+
- GitHub account
- GitHub personal access token with the permissions required by the package for repository and workflow operations

## Installation

```bash
python -m pip install gh-ai-runner
```

## Quick Start

```python
from gh_ai_runner import ai_call

result = ai_call(
    github_token="YOUR_GITHUB_TOKEN",
    prompt="Explain recursion in simple terms.",
)

print(result)
```

The first invocation provisions the configured inference repository and model cache. Later calls reuse the existing setup.

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `github_token` | — | GitHub token used for repository/workflow operations |
| `prompt` | — | Prompt sent to the model |
| `model` | `tinyllama` | Model configuration key |
| `system` | Helpful-assistant prompt | System message |
| `max_tokens` | `512` | Maximum generated tokens |
| `temperature` | `0.7` | Sampling temperature |
| `cache` | `True` | Reuse downloaded model weights |
| `n_ctx` | Model default | Context window override |
| `repo_name` | `ai-inference-runner` | Worker repository name |
| `verbose` | `True` | Emit progress logs |

## Models

The package supports configured GGUF models including TinyLlama, Llama, Phi, Qwen, Gemma, and DeepSeek variants. Available keys depend on the installed package version.

## Architecture

```text
Python client
    ↓
GitHub repository/workflow
    ↓
GitHub Actions runner
    ↓
llama-cpp-python + GGUF model
    ↓
workflow artifact/output
    ↓
Python result
```

## Security

Do not hard-code GitHub tokens. Store credentials in environment variables or a secrets manager and grant only the permissions required by the workflow.

## License

MIT

## Support

Use the repository's GitHub Issues page for bug reports and feature requests.
