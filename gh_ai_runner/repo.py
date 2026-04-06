import base64
import time

import requests

from .logger import _log
from .runner import INFERENCE_SCRIPT, WORKFLOW_YAML

API = "https://api.github.com"


def _headers(token):
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _get_username(token):
    r = requests.get(f"{API}/user", headers=_headers(token))
    r.raise_for_status()
    return r.json()["login"]


def _repo_exists(token, username, repo_name):
    return requests.get(
        f"{API}/repos/{username}/{repo_name}", headers=_headers(token)
    ).status_code == 200


def _commit_file(token, username, repo_name, path, content, message):
    url      = f"{API}/repos/{username}/{repo_name}/contents/{path}"
    existing = requests.get(url, headers=_headers(token))
    sha      = existing.json().get("sha") if existing.status_code == 200 else None
    body     = {"message": message, "content": base64.b64encode(content.encode()).decode()}
    if sha:
        body["sha"] = sha
    requests.put(url, headers=_headers(token), json=body).raise_for_status()


def _wait_for_workflow(token, username, repo_name, timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        r         = requests.get(
            f"{API}/repos/{username}/{repo_name}/actions/workflows",
            headers=_headers(token),
        )
        workflows = r.json().get("workflows", [])
        if any(w["path"] == ".github/workflows/inference.yml" for w in workflows):
            return
        time.sleep(3)
    raise TimeoutError("Workflow never registered.")


def _ensure_repo(token, username, repo_name, verbose):
    if _repo_exists(token, username, repo_name):
        _log("Repo ready", verbose=verbose)
        return

    t = time.time()
    _log("Creating repo...", verbose=verbose)
    r = requests.post(f"{API}/user/repos", headers=_headers(token), json={
        "name": repo_name, "private": False, "auto_init": True,
        "description": "GitHub AI Inference Runner",
    })
    r.raise_for_status()
    time.sleep(2)

    _commit_file(token, username, repo_name,
                 "run_inference.py", INFERENCE_SCRIPT, "Add inference script")
    _commit_file(token, username, repo_name,
                 ".github/workflows/inference.yml", WORKFLOW_YAML, "Add inference workflow")

    _wait_for_workflow(token, username, repo_name)
    _log("Repo created and ready", since=t, verbose=verbose)


def _sync_files(token, username, repo_name, verbose):
    _log("Syncing runner files...", verbose=verbose)
    _commit_file(token, username, repo_name,
                 "run_inference.py", INFERENCE_SCRIPT, "Sync inference script")
    _commit_file(token, username, repo_name,
                 ".github/workflows/inference.yml", WORKFLOW_YAML, "Sync workflow")
    _log("Runner files synced", verbose=verbose)