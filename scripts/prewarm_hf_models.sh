#!/usr/bin/env bash
set -euo pipefail

# Pre-downloads critical HuggingFace assets used by SceneSmith workers so
# runtime jobs don't fail on transient network errors.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p data/hf_cache

echo "[prewarm] Starting HuggingFace model prewarm..."
echo "[prewarm] Cache dir: ${ROOT_DIR}/data/hf_cache"

docker compose run --rm -T \
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

from huggingface_hub import hf_hub_download, snapshot_download

import open_clip

max_attempts = 8
sleep_s = 15

token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def with_retries(label, fn):
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            print(f"[prewarm] {label} attempt {attempt}/{max_attempts} failed: {exc}")
            if attempt == max_attempts:
                raise
            time.sleep(sleep_s)


def prewarm_moge():
    repo_id = "Ruicheng/moge-vitl"
    local_dir = "/app/data/hf_cache/prewarm/moge-vitl"
    print(f"[prewarm] repo={repo_id}")
    print(f"[prewarm] local_dir={local_dir}")
    with_retries(
        "moge snapshot_download",
        lambda: snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            token=token,
        ),
    )
    print("[prewarm] moge SUCCESS")


def prewarm_dfn5b_openclip():
    repo_id = "apple/DFN5B-CLIP-ViT-H-14-378"
    print(f"[prewarm] repo={repo_id}")
    required_files = [
        "open_clip_config.json",
        "open_clip_pytorch_model.bin",
    ]

    for filename in required_files:
        print(f"[prewarm] downloading {filename}")
        path = with_retries(
            f"dfn5b {filename}",
            lambda: hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=os.getenv("HF_HOME"),
                token=token,
                resume_download=True,
            ),
        )
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"[prewarm] cached {filename}: {path} ({size_mb:.1f} MB)")

    print("[prewarm] validating open_clip model load on cpu")
    with_retries(
        "open_clip load",
        lambda: open_clip.create_model_and_transforms(
            "ViT-H-14-378-quickgelu", pretrained="dfn5b", device="cpu"
        ),
    )
    print("[prewarm] dfn5b open_clip load SUCCESS")


prewarm_moge()
prewarm_dfn5b_openclip()
print("[prewarm] all targets completed")
PY

echo "[prewarm] Completed."
