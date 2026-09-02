from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from .types import (
    InferenceResult,
    JobCancelledError,
    JobFailedError,
    JobStatus,
    RetryPolicy,
)

if TYPE_CHECKING:
    from .client import GitHubAIRunner


class AIJob:
    """A remotely executing inference job.

    Jobs are lightweight handles and can be stored or reconstructed with
    :meth:`GitHubAIRunner.get_job` in another process.
    """

    def __init__(
        self,
        runner: GitHubAIRunner,
        job_id: str,
        request: dict,
        retry_policy: RetryPolicy,
        *,
        attempt: int = 1,
        created_at: Optional[datetime] = None,
    ) -> None:
        self.runner = runner
        self.job_id = job_id
        self.request = dict(request)
        self.retry_policy = retry_policy
        self.attempt = attempt
        self.created_at = created_at or datetime.now(timezone.utc)
        self.run_id: Optional[int] = None
        self._cancelled = False
        self._result: Optional[InferenceResult] = None
        self._lock = threading.RLock()

    @property
    def correlation_id(self) -> str:
        return f"{self.job_id}-{self.attempt}"

    @property
    def actions_url(self) -> Optional[str]:
        if self.run_id is None:
            return None
        return f"https://github.com/{self.runner.username}/{self.runner.repo_name}/actions/runs/{self.run_id}"

    def refresh(self) -> JobStatus:
        with self._lock:
            if self._result is not None:
                return JobStatus.COMPLETED
            run = self.runner._find_run(self, wait=False)
            if run is None:
                return JobStatus.CANCELLED if self._cancelled else JobStatus.DISPATCHED
            self.run_id = int(run["id"])
            status = run.get("status", "queued")
            if status != "completed":
                return JobStatus.IN_PROGRESS if status == "in_progress" else JobStatus.QUEUED
            conclusion = run.get("conclusion")
            if conclusion == "success":
                return JobStatus.COMPLETED
            if conclusion == "cancelled":
                return JobStatus.CANCELLED
            return JobStatus.FAILED

    def wait(self, timeout: Optional[float] = None) -> AIJob:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self._cancelled:
                raise JobCancelledError(f"Job {self.job_id} was cancelled", run_id=self.run_id)
            try:
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                run = self.runner._find_run(self, wait=True, timeout=remaining)
                self.run_id = int(run["id"])
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                completed = self.runner._wait_for_completion(self, timeout=remaining)
                conclusion = completed.get("conclusion")
                if self._cancelled:
                    raise JobCancelledError(
                        f"Job {self.job_id} was cancelled", run_id=self.run_id, conclusion=conclusion
                    )
                if conclusion == "success":
                    return self
                if conclusion == "cancelled" or self._cancelled:
                    raise JobCancelledError(
                        f"Job {self.job_id} was cancelled", run_id=self.run_id, conclusion=conclusion
                    )
                raise JobFailedError(
                    f"Workflow failed ({conclusion}) — {self.actions_url}",
                    run_id=self.run_id,
                    conclusion=conclusion,
                )
            except (JobCancelledError, TimeoutError):
                raise
            except JobFailedError:
                if self.attempt > self.retry_policy.max_retries or self._cancelled:
                    raise
                if self.retry_policy.delay_seconds:
                    time.sleep(self.retry_policy.delay_seconds)
                self.attempt += 1
                self.run_id = None
                self.runner._dispatch(self)

    def result(self, timeout: Optional[float] = None) -> InferenceResult:
        with self._lock:
            if self._result is not None:
                return self._result
        self.wait(timeout=timeout)
        downloaded = self.runner._download_result(self)
        with self._lock:
            if self._result is None:
                self._result = downloaded
            return self._result

    def cancel(self) -> bool:
        with self._lock:
            run = self.runner._find_run(self, wait=True, timeout=self.runner.run_discovery_timeout)
            self.run_id = int(run["id"])
            if run.get("status") == "completed":
                return False
            cancelled = self.runner._cancel_run(self.run_id)
            self._cancelled = cancelled
            return cancelled
