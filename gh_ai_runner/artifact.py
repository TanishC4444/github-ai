import io
import json
import time
import zipfile

import requests

from .integrity import text_sha256
from .logger import _log
from .repo import API, _headers
from .types import (
    InferenceResult,
    IntegrityError,
    IntegrityInfo,
    ResourceUsage,
    TokenUsage,
)


def _download_output(token, username, repo_name, run_id, verbose, timeout=60):
    start  = time.time()
    target = None

    while time.time() - start < timeout:
        r         = requests.get(
            f"{API}/repos/{username}/{repo_name}/actions/runs/{run_id}/artifacts",
            headers=_headers(token),
        )
        artifacts = r.json().get("artifacts", [])
        target    = next((a for a in artifacts if a["name"] == "ai-output"), None)
        if target:
            break
        _log("Waiting for artifact...", verbose=verbose)
        time.sleep(5)

    if not target:
        raise RuntimeError("No ai-output artifact found after waiting.")

    r = requests.get(
        f"{API}/repos/{username}/{repo_name}/actions/artifacts/{target['id']}/zip",
        headers=_headers(token),
        allow_redirects=False,
    )
    location  = r.headers.get("Location", "")
    zip_bytes = requests.get(location).content if location else r.content

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        _log(f"Artifact contents: {z.namelist()}", verbose=verbose)
        name = next((n for n in z.namelist() if n.endswith("output.txt")), None)
        if not name:
            raise RuntimeError(f"output.txt not found in artifact. Contents: {z.namelist()}")
        with z.open(name) as f:
            return f.read().decode()


def _download_result(
    token,
    username,
    repo_name,
    run_id,
    job_id,
    correlation_id,
    verbose,
    timeout=60,
    expected_request_sha256=None,
):
    start = time.monotonic()
    target = None
    artifact_name = f"ai-output-{correlation_id}"
    while time.monotonic() - start < timeout:
        response = requests.get(
            f"{API}/repos/{username}/{repo_name}/actions/runs/{run_id}/artifacts",
            headers=_headers(token),
        )
        response.raise_for_status()
        target = next(
            (item for item in response.json().get("artifacts", []) if item["name"] == artifact_name),
            None,
        )
        if target:
            break
        time.sleep(2)
    if not target:
        raise RuntimeError(f"No {artifact_name} artifact found after waiting.")

    response = requests.get(
        f"{API}/repos/{username}/{repo_name}/actions/artifacts/{target['id']}/zip",
        headers=_headers(token),
        allow_redirects=True,
    )
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        name = next((item for item in archive.namelist() if item.endswith("result.json")), None)
        if not name:
            raise RuntimeError(f"result.json not found in artifact. Contents: {archive.namelist()}")
        data = json.loads(archive.read(name).decode("utf-8"))

    if data.get("job_id") != job_id or data.get("correlation_id") != correlation_id:
        raise IntegrityError("Result identity does not match the requested job", run_id=run_id)
    integrity = data.get("integrity") or {}
    if integrity and integrity.get("algorithm") != "sha256":
        raise IntegrityError("Result uses an unsupported integrity algorithm", run_id=run_id)
    actual_output_sha256 = text_sha256(data.get("output", ""))
    if integrity and integrity.get("output_sha256") != actual_output_sha256:
        raise IntegrityError("Result output checksum verification failed", run_id=run_id)
    if expected_request_sha256 and integrity.get("request_sha256") != expected_request_sha256:
        raise IntegrityError("Result request checksum verification failed", run_id=run_id)

    usage = data.get("usage") or {}
    resources = data.get("resources") or {}
    return InferenceResult(
        job_id=job_id,
        correlation_id=correlation_id,
        run_id=run_id,
        output=data.get("output", ""),
        model=data.get("model", ""),
        provider=data.get("provider", "local"),
        usage=TokenUsage(
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            total_tokens=int(usage.get("total_tokens", 0)),
        ),
        resources=ResourceUsage(**{
            field: resources.get(field, default)
            for field, default in {
                "cpu_count": 0,
                "memory_total_bytes": 0,
                "memory_available_bytes": 0,
                "disk_total_bytes": 0,
                "disk_free_bytes": 0,
                "peak_memory_bytes": 0,
                "duration_seconds": 0.0,
                "runner_os": "",
            }.items()
        }),
        integrity=IntegrityInfo(
            algorithm=integrity.get("algorithm", "sha256"),
            request_sha256=integrity.get("request_sha256", ""),
            output_sha256=integrity.get("output_sha256", ""),
        ),
        raw=data,
    )
