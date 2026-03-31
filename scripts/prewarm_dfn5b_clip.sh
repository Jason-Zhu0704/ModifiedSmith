#!/usr/bin/env bash
set -euo pipefail

# Robust prewarm for OpenCLIP DFN5B weights used by HSSD retrieval.
# Runs entirely inside Docker and stores cache in ./data/hf_cache (host-mounted).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p data/hf_cache data/torch_cache

MAX_ATTEMPTS="${MAX_ATTEMPTS:-20}"
ATTEMPT_TIMEOUT_SEC="${ATTEMPT_TIMEOUT_SEC:-1800}"
SLEEP_SEC="${SLEEP_SEC:-20}"
MIN_BYTES="${MIN_BYTES:-3000000000}" # ~3GB minimum sanity check

echo "[dfn5b] start prewarm"
echo "[dfn5b] cache=${ROOT_DIR}/data/hf_cache"
echo "[dfn5b] max_attempts=${MAX_ATTEMPTS} timeout=${ATTEMPT_TIMEOUT_SEC}s"

attempt=1
while [[ "${attempt}" -le "${MAX_ATTEMPTS}" ]]; do
  echo "[dfn5b] attempt ${attempt}/${MAX_ATTEMPTS}"
  if timeout "${ATTEMPT_TIMEOUT_SEC}" docker compose run --rm -T \
    -e HF_HOME=/app/data/hf_cache \
    -e HUGGINGFACE_HUB_CACHE=/app/data/hf_cache/hub \
    -e TRANSFORMERS_CACHE=/app/data/hf_cache/transformers \
    -e TORCH_HOME=/app/data/torch_cache \
    -e HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}" \
    -e HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-1200}" \
    -e HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
    -e MIN_BYTES="${MIN_BYTES}" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}" \
    scenesmith \
    /app/.venv/bin/python -u - <<'PY'
import os

from huggingface_hub import hf_hub_download
import open_clip

repo_id = "apple/DFN5B-CLIP-ViT-H-14-378"
filename = "open_clip_pytorch_model.bin"
min_bytes = int(os.getenv("MIN_BYTES", "3000000000"))
token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

print(f"[dfn5b] download {repo_id}/{filename}")
path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    cache_dir=os.getenv("HF_HOME"),
    token=token,
    resume_download=True,
)
size = os.path.getsize(path)
print(f"[dfn5b] cached file: {path}")
print(f"[dfn5b] size_bytes={size}")
if size < min_bytes:
    raise RuntimeError(f"file too small: {size} < {min_bytes}")

print("[dfn5b] validating open_clip load")
open_clip.create_model_and_transforms(
    "ViT-H-14-378-quickgelu", pretrained="dfn5b", device="cpu"
)
print("[dfn5b] SUCCESS")
PY
  then
    echo "[dfn5b] completed"
    exit 0
  fi

  echo "[dfn5b] attempt ${attempt} failed or timed out; sleep ${SLEEP_SEC}s"
  sleep "${SLEEP_SEC}"
  attempt=$((attempt + 1))
done

echo "[dfn5b] failed after ${MAX_ATTEMPTS} attempts"
exit 1
