import time

import requests

from .logger import _log
from .repo import API, _headers


def _find_run(token, username, repo_name, correlation_id, verbose, branch="main", created_at=None, timeout=180):
    """Find the one workflow run whose explicit run title contains correlation_id."""
    deadline = time.monotonic() + max(0, timeout)
    first = True
    while first or time.monotonic() < deadline:
        first = False
        r = requests.get(
            f"{API}/repos/{username}/{repo_name}/actions/workflows/inference.yml/runs",
            headers=_headers(token),
            params={"event": "workflow_dispatch", "branch": branch, "per_page": 100},
        )
        r.raise_for_status()
        for run in r.json().get("workflow_runs", []):
            if run.get("display_title") == f"AI inference {correlation_id}":
                _log(f"Runner picked up job (run #{run['id']})", verbose=verbose)
                return run
        if timeout <= 0:
            return None
        time.sleep(min(2, max(0, deadline - time.monotonic())))
    raise TimeoutError(f"Timed out waiting for workflow run {correlation_id} to appear.")


def _wait_for_completion(token, username, repo_name, run_id, verbose, timeout=900, poll=8):
    # poll reduced from 12s to 8s — shaves ~30s off a typical cached run
    start       = time.time()
    last_status = None
    while time.time() - start < timeout:
        r          = requests.get(
            f"{API}/repos/{username}/{repo_name}/actions/runs/{run_id}",
            headers=_headers(token),
        )
        data       = r.json()
        status     = data["status"]
        conclusion = data.get("conclusion")

        if status != last_status:
            _log(f"Runner: {status}{' -> ' + conclusion if conclusion else ''}",
                 verbose=verbose)
            last_status = status

        if status == "completed":
            return data

        time.sleep(poll)
    raise TimeoutError("Timed out waiting for workflow to complete.")


# Backwards-compatible internal aliases retained for integrations that imported
# the old helpers. New code uses explicit correlation IDs instead.
def _snapshot_run_ids(token, username, repo_name):
    r = requests.get(
        f"{API}/repos/{username}/{repo_name}/actions/runs",
        headers=_headers(token),
        params={"per_page": 20},
    )
    r.raise_for_status()
    return {run["id"] for run in r.json().get("workflow_runs", [])}


def _wait_for_run(token, username, repo_name, seen_ids, verbose, timeout=180):
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        time.sleep(2)
        r = requests.get(
            f"{API}/repos/{username}/{repo_name}/actions/runs",
            headers=_headers(token),
            params={"per_page": 10},
        )
        r.raise_for_status()
        for run in r.json().get("workflow_runs", []):
            if run["id"] not in seen_ids:
                return run["id"]
    raise TimeoutError("Timed out waiting for workflow run to appear.")
