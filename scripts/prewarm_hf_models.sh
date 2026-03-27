#!/usr/bin/env bash
set -euo pipefail

# Pre-downloads critical HuggingFace assets used by geometry workers so
# runtime jobs don't fail on transient network errors.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p data/hf_cache

echo "[prewarm] Starting HuggingFace model prewarm..."
echo "[prewarm] Cache dir: ${ROOT_DIR}/data/hf_cache"

docker compose run --rm \
  -e HF_HOME=/app/data/hf_cache \
  -e HUGGINGFACE_HUB_CACHE=/app/data/hf_cache/hub \
  -e TRANSFORMERS_CACHE=/app/data/hf_cache/transformers \
  -e HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}" \
  -e HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-1200}" \
  -e HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}" \
  scenesmith \
  /app/.venv/bin/python -u - <<'PY'
import os
import time

from huggingface_hub import snapshot_download

repo_id = "Ruicheng/moge-vitl"
local_dir = "/app/data/hf_cache/prewarm/moge-vitl"
max_attempts = 8
sleep_s = 15

print(f"[prewarm] repo={repo_id}")
print(f"[prewarm] local_dir={local_dir}")

for attempt in range(1, max_attempts + 1):
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"),
        )
        print("[prewarm] SUCCESS")
        break
    except Exception as e:
        print(f"[prewarm] attempt {attempt}/{max_attempts} failed: {e}")
        if attempt == max_attempts:
            raise
        time.sleep(sleep_s)
PY

echo "[prewarm] Completed."
