from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class JobStatus(str, Enum):
    DISPATCHED = "dispatched"
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 0
    delay_seconds: float = 5.0

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be at least 0")
        if self.delay_seconds < 0:
            raise ValueError("delay_seconds must be at least 0")


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration for an OpenAI-compatible BYOK endpoint.

    The fixed ``secret_name`` identifies the package's dedicated GitHub Actions
    secret. The key itself is never included in dispatch inputs, logs, or artifacts.
    """

    name: str
    model: str
    base_url: str
    secret_name: str = "GH_AI_PROVIDER_API_KEY"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("provider name cannot be empty")
        if not self.model.strip():
            raise ValueError("provider model cannot be empty")
        if not self.base_url.startswith("https://"):
            raise ValueError("provider base_url must use HTTPS")
        if self.secret_name != "GH_AI_PROVIDER_API_KEY":
            raise ValueError("0.2 uses the dedicated GH_AI_PROVIDER_API_KEY secret slot")

    @classmethod
    def groq(cls, model: str) -> ProviderConfig:
        return cls("groq", model, "https://api.groq.com/openai/v1")

    @classmethod
    def openai(cls, model: str) -> ProviderConfig:
        return cls("openai", model, "https://api.openai.com/v1")


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class ResourceUsage:
    cpu_count: int = 0
    memory_total_bytes: int = 0
    memory_available_bytes: int = 0
    disk_total_bytes: int = 0
    disk_free_bytes: int = 0
    peak_memory_bytes: int = 0
    duration_seconds: float = 0.0
    runner_os: str = ""


@dataclass(frozen=True)
class InferenceResult:
    job_id: str
    correlation_id: str
    run_id: int
    output: str
    model: str
    provider: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    resources: ResourceUsage = field(default_factory=ResourceUsage)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("raw", None)
        return value


class JobFailedError(RuntimeError):
    def __init__(self, message: str, *, run_id: Optional[int] = None, conclusion: Optional[str] = None):
        super().__init__(message)
        self.run_id = run_id
        self.conclusion = conclusion


class JobCancelledError(JobFailedError):
    pass
