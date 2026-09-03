from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

from .artifact import _download_result
from .batch import BatchJob, ComparisonJob
from .job import AIJob
from .logger import _log
from .metadata import DEFAULT_METADATA_PATH, MetadataStore, now_iso, result_fields
from .models import MODELS
from .polling import _find_run, _wait_for_completion
from .repo import (
    API,
    _ensure_repo,
    _get_default_branch,
    _get_username,
    _headers,
    _set_actions_secret,
    _sync_files,
)
from .types import (
    InferenceRequest,
    InferenceResult,
    JobRecord,
    JobStatus,
    ProviderConfig,
    RetryPolicy,
)
from .validation import _validate


class GitHubAIRunner:
    """Client for dispatching and managing ephemeral GitHub Actions AI jobs."""

    def __init__(
        self,
        github_token: str,
        *,
        repo_name: str = "ai-inference-runner",
        verbose: bool = True,
        run_discovery_timeout: float = 180,
        completion_timeout: float = 900,
        poll_interval: float = 8,
        metadata_path: Optional[str | Path] = DEFAULT_METADATA_PATH,
    ) -> None:
        if not github_token:
            raise ValueError("github_token cannot be empty")
        self.github_token = github_token
        self.repo_name = repo_name
        self.verbose = verbose
        self.run_discovery_timeout = run_discovery_timeout
        self.completion_timeout = completion_timeout
        self.poll_interval = poll_interval
        self._username: Optional[str] = None
        self._default_branch: Optional[str] = None
        self._prepared = False
        self._prepare_lock = threading.Lock()
        self._metadata = MetadataStore(metadata_path) if metadata_path is not None else None

    @property
    def username(self) -> str:
        if self._username is None:
            self._username = _get_username(self.github_token)
        return self._username

    def prepare(self) -> None:
        if self._prepared:
            return
        with self._prepare_lock:
            if self._prepared:
                return
            _ensure_repo(self.github_token, self.username, self.repo_name, self.verbose)
            _sync_files(self.github_token, self.username, self.repo_name, self.verbose)
            self._default_branch = _get_default_branch(
                self.github_token, self.username, self.repo_name
            )
            self._prepared = True

    @property
    def default_branch(self) -> str:
        self.prepare()
        return self._default_branch or "main"

    def configure_provider(self, provider: ProviderConfig, api_key: str) -> None:
        """Encrypt and store a BYOK key as a GitHub Actions repository secret."""
        if not api_key:
            raise ValueError("api_key cannot be empty")
        self.prepare()
        _set_actions_secret(
            self.github_token,
            self.username,
            self.repo_name,
            provider.secret_name,
            api_key,
        )
        _log(f"Configured provider secret {provider.secret_name}", verbose=self.verbose)

    def submit(
        self,
        prompt: str,
        *,
        model: str = "tinyllama",
        system: str = "You are a helpful assistant.",
        max_tokens: int = 512,
        temperature: float = 0.7,
        cache: bool = True,
        n_ctx: Optional[int] = None,
        provider: Optional[ProviderConfig] = None,
        retry_policy: Optional[RetryPolicy] = None,
        _batch_id: Optional[str] = None,
    ) -> AIJob:
        if not prompt.strip():
            raise ValueError("prompt cannot be empty")
        if provider is None and model not in MODELS:
            raise ValueError(f"model must be one of: {list(MODELS.keys())}")
        _validate(model if provider is None else None, max_tokens, temperature, n_ctx)
        self.prepare()

        request = {
            "prompt": prompt,
            "system": system + " Be concise and direct. Avoid unnecessary filler.",
            "model": provider.model if provider else model,
            "local_model": model,
            "cache": cache,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n_ctx": n_ctx,
            "provider": provider,
        }
        job = AIJob(
            self,
            uuid.uuid4().hex,
            request,
            retry_policy or RetryPolicy(),
            batch_id=_batch_id,
        )
        self._dispatch(job)
        return job

    def submit_batch(self, requests_: list[InferenceRequest]) -> BatchJob:
        """Dispatch a collection of independent requests and return one batch handle."""
        if not requests_:
            raise ValueError("requests_ must contain at least one request")
        for request in requests_:
            if not isinstance(request, InferenceRequest):
                raise TypeError("each batch item must be an InferenceRequest")
            if not request.prompt.strip():
                raise ValueError("batch prompts cannot be empty")
            if request.provider is None and request.model not in MODELS:
                raise ValueError(f"model must be one of: {list(MODELS.keys())}")
            _validate(
                request.model if request.provider is None else None,
                request.max_tokens,
                request.temperature,
                request.n_ctx,
            )
        batch_id = uuid.uuid4().hex
        jobs = []
        for request in requests_:
            jobs.append(
                self.submit(
                    request.prompt,
                    model=request.model,
                    system=request.system,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    cache=request.cache,
                    n_ctx=request.n_ctx,
                    provider=request.provider,
                    retry_policy=request.retry_policy,
                    _batch_id=batch_id,
                )
            )
        return BatchJob(jobs, batch_id=batch_id)

    def compare(self, prompt: str, models: list[str], **kwargs) -> ComparisonJob:
        """Dispatch the same prompt to distinct local models for side-by-side results."""
        if len(models) < 2:
            raise ValueError("compare requires at least two models")
        if len(set(models)) != len(models):
            raise ValueError("comparison models must be unique")
        if kwargs.get("provider") is not None:
            raise ValueError("compare currently supports local models; use submit_batch for providers")
        invalid = [model for model in models if model not in MODELS]
        if invalid:
            raise ValueError(f"unknown comparison models: {invalid}")
        requests_ = [InferenceRequest(prompt=prompt, model=model, **kwargs) for model in models]
        batch = self.submit_batch(requests_)
        return ComparisonJob(batch.jobs, models, batch_id=batch.batch_id)

    def list_jobs(self, limit: int = 100) -> list[JobRecord]:
        if self._metadata is None:
            return []
        return self._metadata.list(limit)

    def resume(self, job_id: str) -> AIJob:
        if self._metadata is None:
            raise RuntimeError("metadata persistence is disabled")
        record = self._metadata.get(job_id)
        if record is None:
            raise KeyError(f"Unknown job {job_id}")
        if record.owner != self.username or record.repo_name != self.repo_name:
            raise ValueError(
                f"Job {job_id} belongs to {record.owner}/{record.repo_name}, "
                f"not {self.username}/{self.repo_name}"
            )
        return AIJob(
            self,
            record.job_id,
            {},
            RetryPolicy(),
            attempt=record.attempt,
            created_at=datetime.fromisoformat(record.created_at),
            request_digest=record.request_sha256,
            prompt_digest=record.prompt_sha256,
            batch_id=record.batch_id,
        )

    def get_job(
        self,
        job_id: str,
        *,
        attempt: int = 1,
        created_at: Optional[datetime] = None,
    ) -> AIJob:
        """Reconstruct a job handle for status checks or cancellation."""
        self.prepare()
        return AIJob(
            self,
            job_id,
            {},
            RetryPolicy(),
            attempt=attempt,
            created_at=created_at or datetime.fromtimestamp(0, timezone.utc),
        )

    def _record_job(
        self,
        job: AIJob,
        status: JobStatus,
        *,
        result: Optional[InferenceResult] = None,
        error: Optional[str] = None,
    ) -> None:
        if self._metadata is None:
            return
        existing = self._metadata.get(job.job_id, job.attempt)
        request = job.request
        model = request.get("model") or (existing.model if existing else "unknown")
        provider_config = request.get("provider")
        provider = (
            provider_config.name
            if provider_config
            else (existing.provider if existing else "local")
        )
        fields = result_fields(result)
        record = JobRecord(
            correlation_id=job.correlation_id,
            job_id=job.job_id,
            attempt=job.attempt,
            batch_id=job.batch_id,
            owner=self.username,
            repo_name=self.repo_name,
            status=status.value,
            run_id=job.run_id,
            model=model,
            provider=provider,
            prompt_sha256=job.prompt_digest or (existing.prompt_sha256 if existing else ""),
            request_sha256=job.request_digest or (existing.request_sha256 if existing else ""),
            output_sha256=fields.get("output_sha256") or (existing.output_sha256 if existing else None),
            prompt_tokens=fields.get("prompt_tokens") if result else (existing.prompt_tokens if existing else None),
            completion_tokens=fields.get("completion_tokens") if result else (existing.completion_tokens if existing else None),
            total_tokens=fields.get("total_tokens") if result else (existing.total_tokens if existing else None),
            duration_seconds=fields.get("duration_seconds") if result else (existing.duration_seconds if existing else None),
            created_at=existing.created_at if existing else job.created_at.isoformat(),
            updated_at=now_iso(),
            error=error,
        )
        self._metadata.upsert(record)

    def _dispatch(self, job: AIJob) -> None:
        provider = job.request.get("provider")
        config = {
            "model": job.request["model"],
            "max_tokens": job.request["max_tokens"],
            "temperature": job.request["temperature"],
            "n_ctx": job.request["n_ctx"],
            "provider_name": provider.name if provider else "local",
            "provider_url": provider.base_url if provider else "",
        }
        inputs = {
            "job_id": job.job_id,
            "attempt": str(job.attempt),
            "prompt": job.request["prompt"],
            "system": job.request["system"],
            "backend": "provider" if provider else "local",
            "local_model": job.request["local_model"],
            "cache": "true" if job.request["cache"] else "false",
            "config": json.dumps(config, separators=(",", ":")),
        }
        if sum(len(key) + len(value) for key, value in inputs.items()) > 65_000:
            raise ValueError("Combined workflow inputs exceed GitHub's 65,535 character limit")
        _log(f"Dispatching job {job.job_id} attempt {job.attempt}", verbose=self.verbose)
        response = requests.post(
            f"{API}/repos/{self.username}/{self.repo_name}/actions/workflows/inference.yml/dispatches",
            headers=_headers(self.github_token),
            json={"ref": self.default_branch, "inputs": inputs},
            timeout=30,
        )
        response.raise_for_status()
        self._record_job(job, JobStatus.DISPATCHED)

    def _find_run(self, job: AIJob, *, wait: bool, timeout: Optional[float] = None):
        run = _find_run(
            self.github_token,
            self.username,
            self.repo_name,
            job.correlation_id,
            self.verbose,
            branch=self.default_branch,
            created_at=job.created_at,
            timeout=(self.run_discovery_timeout if timeout is None else timeout) if wait else 0,
        )
        if wait and run is None:
            raise TimeoutError(f"Timed out waiting for workflow run {job.correlation_id} to appear.")
        return run

    def _wait_for_completion(self, job: AIJob, timeout: Optional[float] = None):
        return _wait_for_completion(
            self.github_token,
            self.username,
            self.repo_name,
            job.run_id,
            self.verbose,
            timeout=self.completion_timeout if timeout is None else timeout,
            poll=self.poll_interval,
        )

    def _download_result(self, job: AIJob) -> InferenceResult:
        return _download_result(
            self.github_token,
            self.username,
            self.repo_name,
            job.run_id,
            job.job_id,
            job.correlation_id,
            self.verbose,
            expected_request_sha256=job.request_digest or None,
        )

    def _cancel_run(self, run_id: int) -> bool:
        response = requests.post(
            f"{API}/repos/{self.username}/{self.repo_name}/actions/runs/{run_id}/cancel",
            headers=_headers(self.github_token),
            timeout=30,
        )
        if response.status_code == 409:
            return False
        response.raise_for_status()
        return True
