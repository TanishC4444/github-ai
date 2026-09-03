# Changelog

## 0.3.0

- Add concurrent batch submission and collection with `InferenceRequest` and `BatchJob`.
- Add side-by-side local model comparisons with `ComparisonJob`.
- Add canonical request and output SHA-256 verification.
- Add SQLite job and batch metadata, history listing, and cross-process job resume.
- Exclude prompts, outputs, GitHub tokens, and provider keys from persistent metadata.

## 0.2.0

- Add asynchronous `AIJob` handles with status, waiting, cancellation, and retries.
- Add exact workflow correlation and job-specific structured artifacts.
- Add token and runner resource telemetry.
- Add encrypted BYOK setup for OpenAI-compatible HTTPS providers.
- Preserve the blocking `ai_call()` API.
