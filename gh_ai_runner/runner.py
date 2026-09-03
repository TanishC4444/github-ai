import hashlib

INFERENCE_SCRIPT = r'''
import hashlib
import json
import os
import platform
import resource
import shutil
import time
import urllib.error
import urllib.request
import warnings

warnings.filterwarnings("ignore")

MODEL_MAP = {
    "tinyllama": {
        "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", "n_ctx": 2048,
    },
    "llama": {
        "url": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "filename": "Llama-3.2-1B-Instruct-Q4_K_M.gguf", "n_ctx": 4096,
    },
    "phi3": {
        "url": "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf",
        "filename": "Phi-3.5-mini-instruct-Q4_K_M.gguf", "n_ctx": 4096,
    },
    "qwen": {
        "url": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf", "n_ctx": 4096,
    },
    "gemma2": {
        "url": "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf",
        "filename": "gemma-2-2b-it-Q4_K_M.gguf", "n_ctx": 4096,
    },
    "deepseek": {
        "url": "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
        "filename": "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf", "n_ctx": 4096,
    },
}


def memory_info():
    values = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as stream:
            for line in stream:
                key, value = line.split(":", 1)
                values[key] = int(value.strip().split()[0]) * 1024
    except (OSError, ValueError):
        pass
    return values


def resource_snapshot(started):
    memory = memory_info()
    disk = shutil.disk_usage(".")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() != "Darwin":
        peak *= 1024
    return {
        "cpu_count": os.cpu_count() or 1,
        "memory_total_bytes": memory.get("MemTotal", 0),
        "memory_available_bytes": memory.get("MemAvailable", 0),
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "peak_memory_bytes": peak,
        "duration_seconds": round(time.monotonic() - started, 3),
        "runner_os": platform.platform(),
    }


def provider_inference(messages, max_tokens, temperature, model, base_url):
    base_url = base_url.rstrip("/")
    api_key = os.environ.get("PROVIDER_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "GitHub Actions secret 'GH_AI_PROVIDER_API_KEY' is missing"
        )
    payload = json.dumps({
        "model": model, "messages": messages,
        "max_tokens": max_tokens, "temperature": temperature,
    }).encode()
    request = urllib.request.Request(
        base_url + "/chat/completions", data=payload,
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:1000]
        raise RuntimeError(f"Provider returned HTTP {error.code}: {detail}") from error


def local_inference(messages, max_tokens, temperature, local_model, requested_n_ctx):
    from llama_cpp import Llama

    cfg = MODEL_MAP[local_model]
    n_ctx = int(requested_n_ctx) if requested_n_ctx else cfg["n_ctx"]
    model_path = os.path.join("model_cache", cfg["filename"])
    if not os.path.exists(model_path):
        os.makedirs("model_cache", exist_ok=True)
        urllib.request.urlretrieve(cfg["url"], model_path)
    cpu_count = os.cpu_count() or 1
    threads = max(1, cpu_count - 1) if cpu_count > 2 else cpu_count
    llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=threads, verbose=False)
    return llm.create_chat_completion(
        messages=messages, max_tokens=max_tokens, temperature=temperature
    )


started = time.monotonic()
job_id = os.environ["JOB_ID"]
attempt = int(os.environ.get("ATTEMPT", "1"))
backend = os.environ.get("BACKEND", "local")
config = json.loads(os.environ.get("CONFIG_JSON", "{}"))
provider_name = config.get("provider_name", "local")
model = config["model"]
messages = [
    {"role": "system", "content": os.environ.get("SYSTEM", "You are a helpful assistant.")},
    {"role": "user", "content": os.environ["PROMPT"]},
]
max_tokens = int(config.get("max_tokens", 512))
temperature = float(config.get("temperature", 0.7))
request_document = {
    "prompt": os.environ["PROMPT"],
    "system": os.environ.get("SYSTEM", "You are a helpful assistant."),
    "backend": backend,
    "model": model,
    "local_model": os.environ.get("LOCAL_MODEL", "tinyllama"),
    "cache": os.environ.get("CACHE", "true") == "true",
    "max_tokens": max_tokens,
    "temperature": temperature,
    "n_ctx": config.get("n_ctx"),
    "provider_name": provider_name,
    "provider_url": config.get("provider_url", ""),
}
request_sha256 = hashlib.sha256(
    json.dumps(request_document, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()

result = {
    "schema_version": 1, "job_id": job_id, "attempt": attempt,
    "correlation_id": f"{job_id}-{attempt}", "model": model,
    "provider": provider_name if backend == "provider" else "local",
    "output": "", "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    "resources": {},
    "integrity": {
        "algorithm": "sha256", "request_sha256": request_sha256, "output_sha256": "",
    },
}

try:
    response = (
        provider_inference(
            messages, max_tokens, temperature, model, config.get("provider_url", "")
        )
        if backend == "provider"
        else local_inference(
            messages, max_tokens, temperature,
            os.environ.get("LOCAL_MODEL", "tinyllama"), config.get("n_ctx"),
        )
    )
    result["output"] = response["choices"][0]["message"]["content"] or ""
    usage = response.get("usage") or {}
    result["usage"] = {
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)),
    }
except Exception as error:
    result["error"] = {"type": type(error).__name__, "message": str(error)}
    raise
finally:
    result["integrity"]["output_sha256"] = hashlib.sha256(result["output"].encode()).hexdigest()
    result["resources"] = resource_snapshot(started)
    with open("result.json", "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
'''


WORKFLOW_YAML = """\
name: AI Inference
run-name: AI inference ${{ inputs.job_id }}-${{ inputs.attempt }}
on:
  workflow_dispatch:
    inputs:
      job_id:          { description: "Correlation ID", required: true }
      attempt:         { description: "Attempt number", required: true, default: "1" }
      prompt:          { description: "User prompt", required: true }
      system:          { description: "System prompt", required: false, default: "You are a helpful assistant." }
      backend:         { description: "local or provider", required: true, default: "local" }
      local_model:     { description: "Local model key", required: false, default: "tinyllama" }
      cache:           { description: "Cache weights", required: false, default: "true" }
      config:          { description: "Non-secret inference settings JSON", required: true }

jobs:
  inference:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 1

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Cache venv
        id: cache-venv
        if: ${{ inputs.backend == 'local' }}
        uses: actions/cache@v4
        with:
          path: .venv
          key: venv-llama-cpp-v2

      - name: Cache model weights
        if: ${{ inputs.backend == 'local' && inputs.cache == 'true' }}
        uses: actions/cache@v4
        with:
          path: model_cache
          key: gguf-${{ inputs.local_model }}-v2

      - name: Install local inference dependencies
        if: ${{ inputs.backend == 'local' && steps.cache-venv.outputs.cache-hit != 'true' }}
        run: |
          python -m venv .venv
          CMAKE_ARGS="-DGGML_METAL=off" .venv/bin/pip install llama-cpp-python -q

      - name: Run inference
        env:
          JOB_ID:           ${{ inputs.job_id }}
          ATTEMPT:          ${{ inputs.attempt }}
          PROMPT:           ${{ inputs.prompt }}
          SYSTEM:           ${{ inputs.system }}
          BACKEND:          ${{ inputs.backend }}
          LOCAL_MODEL:      ${{ inputs.local_model }}
          CACHE:            ${{ inputs.cache }}
          CONFIG_JSON:      ${{ inputs.config }}
          PROVIDER_API_KEY: ${{ secrets.GH_AI_PROVIDER_API_KEY }}
        run: |
          if [ "$BACKEND" = "local" ]; then
            .venv/bin/python run_inference.py
          else
            python run_inference.py
          fi

      - name: Upload structured result
        if: ${{ always() && hashFiles('result.json') != '' }}
        uses: actions/upload-artifact@v4
        with:
          name: ai-output-${{ inputs.job_id }}-${{ inputs.attempt }}
          path: result.json
          retention-days: 3
"""


def _script_hash():
    return hashlib.sha256(INFERENCE_SCRIPT.encode()).hexdigest()


def _workflow_hash():
    return hashlib.sha256(WORKFLOW_YAML.encode()).hexdigest()
