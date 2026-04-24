#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from asset_retrieval.asset_retrieval_core.normalization import (
    load_normalization_config,
    normalize_assets_stream,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize HSSD description JSONL into structured asset_index.jsonl"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/data/scenesmith/data/preprocessed/hssd_asset_descriptions.jsonl"),
        help="Path to raw description JSONL",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/data/scenesmith/asset_retrieval/data/processed/asset_index.jsonl"
        ),
        help="Path to normalized output JSONL",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("/data/scenesmith/asset_retrieval/configs"),
        help="Directory containing yaml configs",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "/data/scenesmith/asset_retrieval/data/processed/asset_index_summary.json"
        ),
        help="Optional summary JSON output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_normalization_config(args.config_dir)
    summary = normalize_assets_stream(args.input, args.output, config)

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
