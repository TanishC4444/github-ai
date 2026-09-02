from __future__ import annotations

from typing import Optional

from .client import GitHubAIRunner
from .types import RetryPolicy


def ai_call(
    github_token: str,
    prompt:       str,
    model:        str   = "tinyllama",
    system:       str   = "You are a helpful assistant.",
    max_tokens:   int   = 512,
    temperature:  float = 0.7,
    cache:        bool  = True,
    n_ctx:        Optional[int] = None,
    repo_name:    str   = "ai-inference-runner",
    verbose:      bool  = True,
) -> str:
    """
    Run AI inference on GitHub Actions and return the model output.

    Required:
        github_token: Your GitHub personal access token (needs repo + workflow scopes).
        prompt:       The message / question to send to the model.

    Optional:
        model:        "tinyllama" (default) or "llama".
        system:       System prompt. Default: "You are a helpful assistant."
        max_tokens:   Max tokens to generate. Default: 512. Hard limit: 4096.
        temperature:  0.0 = deterministic, 2.0 = very creative. Default: 0.7.
        cache:        Cache model weights between runs. Default: True.
        n_ctx:        Context window size. Default: 2048 (tinyllama) / 4096 (llama). Max: 8192.
        repo_name:    GitHub repo to create/reuse. Default: "ai-inference-runner".
        verbose:      Print detailed logs. Default: True. Set False for silent mode.

    Returns:
        Model response as a string.

    Raises:
        ValueError: If any parameter is out of safe range.
    """
    runner = GitHubAIRunner(github_token, repo_name=repo_name, verbose=verbose)
    job = runner.submit(
        prompt,
        model=model,
        system=system,
        max_tokens=max_tokens,
        temperature=temperature,
        cache=cache,
        n_ctx=n_ctx,
        retry_policy=RetryPolicy(),
    )
    return job.result().output
