#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPTS_HSSD = REPO_ROOT / "scripts" / "hssd"
if str(SCRIPTS_HSSD) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_HSSD))

from asset_retrieval.asset_retrieval_core.normalization import (  # noqa: E402
    load_normalization_config,
    normalize_category,
)
from description_retrieval_cleaner import (  # noqa: E402
    build_description_retrieval,
    evaluate_retrieval_description_quality,
    remove_forbidden_terms,
)

CATEGORY_TO_DEFAULT_SYNSET = {
    "bed": "bed.n.01",
    "bunk_bed": "bunk_bed.n.01",
    "nightstand": "nightstand.n.01",
    "desk": "desk.n.01",
    "dining_table": "dining_table.n.01",
    "coffee_table": "coffee_table.n.01",
    "side_table": "end_table.n.01",
    "table": "table.n.02",
    "chair": "chair.n.01",
    "office_chair": "swivel_chair.n.01",
    "armchair": "armchair.n.01",
    "sofa": "sofa.n.01",
    "stool": "stool.n.01",
    "ottoman": "ottoman.n.03",
    "bench": "bench.n.01",
    "dresser": "chest_of_drawers.n.01",
    "wardrobe": "wardrobe.n.01",
    "bookcase": "bookcase.n.01",
    "cabinet": "cabinet.n.01",
    "rug": "rug.n.01",
    "wall_art": "wall_art.n.01",
    "wall_mirror": "wall_mirror.n.01",
    "vase": "vase.n.01",
    "flower_in_vase": "flower_in_vase.n.01",
    "plant": "potted_plant.n.01",
    "throw_pillow": "throw_pillow.n.01",
    "floor_lamp": "floor_lamp.n.01",
    "table_lamp": "table_lamp.n.01",
    "wall_lamp": "wall_lamp.n.01",
    "ceiling_lamp": "ceiling_lamp.n.01",
    "ceiling_fan": "ceiling_fan.n.01",
    "refrigerator": "refrigerator.n.01",
    "oven": "oven.n.01",
    "sink": "sink.n.01",
    "toilet": "toilet.n.02",
    "shower_fixture": "showerhead.n.01",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair failed and synset-missing HSSD description records.")
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path("/data/scenesmith/data/preprocessed/hssd_asset_descriptions.jsonl"),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("/data/scenesmith/data/preprocessed/hssd_asset_descriptions.repaired.jsonl"),
    )
    parser.add_argument(
        "--hssd-index-jsonl",
        type=Path,
        default=Path("/data/scenesmith/data/preprocessed/hssd_asset_index.jsonl"),
    )
    parser.add_argument(
        "--wn-index-json",
        type=Path,
        default=Path("/data/scenesmith/data/preprocessed/hssd_wnsynsetkey_index.json"),
    )
    parser.add_argument(
        "--object-categories-json",
        type=Path,
        default=Path("/data/scenesmith/data/preprocessed/object_categories.json"),
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("/data/scenesmith/asset_retrieval/configs"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("/data/scenesmith/data/preprocessed/hssd_asset_descriptions.repaired.summary.json"),
    )
    return parser.parse_args()


def load_asset_index(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            mid = rec.get("mesh_id") or rec.get("asset_id")
            if mid:
                out[mid] = rec
    return out


def load_wordnet_index(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    synset_by_mesh: dict[str, str] = {}
    name_by_mesh: dict[str, str] = {}
    for synset, entries in raw.items():
        for entry in entries:
            mesh_id = entry.get("id")
            if not mesh_id:
                continue
            synset_by_mesh[mesh_id] = synset
            if entry.get("name"):
                name_by_mesh[mesh_id] = entry["name"]
    return synset_by_mesh, name_by_mesh


def load_synset_buckets(path: Path) -> dict[str, list[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for bucket_name, synsets in raw.items():
        if bucket_name == "available_categories" or not isinstance(synsets, list):
            continue
        for synset in synsets:
            out.setdefault(synset, []).append(bucket_name)
    return out


def build_fallback_description(record: dict[str, Any]) -> str:
    name = record.get("name") or "Unnamed object"
    synset = record.get("synset") or "unknown category object"
    material = record.get("primary_material") or "unknown material"
    style = record.get("style") or "unspecified style"
    color = record.get("color_tone") or "unspecified color tone"
    features = record.get("key_features") or []
    feature_text = ""
    if features:
        feature_text = f" Key features include: {', '.join(str(f) for f in features[:5])}."
    return (
        f"This asset is {name}, categorized as {synset}. "
        f"It has {style}, {color} appearance and is primarily made of {material}."
        f"{feature_text}"
    )


def main() -> None:
    args = parse_args()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    normalization_cfg = load_normalization_config(args.config_dir)
    asset_idx = load_asset_index(args.hssd_index_jsonl)
    synset_by_mesh, wn_name_by_mesh = load_wordnet_index(args.wn_index_json)
    synset_buckets = load_synset_buckets(args.object_categories_json)

    stats = Counter()

    with args.input_jsonl.open("r", encoding="utf-8") as fin, args.output_jsonl.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if not line.strip():
                continue
            stats["total"] += 1
            rec = json.loads(line)
            mesh_id = rec.get("mesh_id") or rec.get("asset_id")
            idx_rec = asset_idx.get(mesh_id, {})

            if rec.get("status") != "success":
                stats["failed_before"] += 1

            if rec.get("synset") is None:
                stats["synset_missing_before"] += 1

            if not rec.get("name"):
                repaired_name = wn_name_by_mesh.get(mesh_id) or idx_rec.get("name")
                if repaired_name:
                    rec["name"] = repaired_name
                    stats["name_filled"] += 1

            if rec.get("synset") is None:
                exact_syn = synset_by_mesh.get(mesh_id) or idx_rec.get("synset")
                if exact_syn:
                    rec["synset"] = exact_syn
                    rec["synset_source"] = "mesh_index"
                    stats["synset_filled_exact"] += 1
                else:
                    inferred_category, conf = normalize_category(rec, normalization_cfg)
                    fallback_synset = CATEGORY_TO_DEFAULT_SYNSET.get(inferred_category)
                    if fallback_synset:
                        rec["synset"] = fallback_synset
                        rec["synset_source"] = "inferred"
                        rec["synset_infer_confidence"] = conf
                        stats["synset_filled_inferred"] += 1

            if not rec.get("category_buckets"):
                synset = rec.get("synset")
                if synset and synset in synset_buckets:
                    rec["category_buckets"] = synset_buckets[synset]
                    stats["category_buckets_filled"] += 1
                elif idx_rec.get("category_buckets"):
                    rec["category_buckets"] = idx_rec["category_buckets"]
                    stats["category_buckets_filled"] += 1

            desc = rec.get("description")
            if not isinstance(desc, str) or not desc.strip():
                rec["description"] = build_fallback_description(rec)
                stats["description_fallback_built"] += 1

            cleaned = remove_forbidden_terms(rec["description"])
            rec["description"] = cleaned
            retrieval = build_description_retrieval(cleaned)
            issues = evaluate_retrieval_description_quality(retrieval)
            rec["description_retrieval"] = retrieval
            rec["description_retrieval_issues"] = issues
            rec["needs_regen"] = bool(issues)
            if issues:
                stats["needs_regen_true"] += 1

            if rec.get("status") != "success":
                has_core = bool(rec.get("name")) and bool(rec.get("synset"))
                if has_core:
                    rec["status"] = "success"
                    rec["repair_source"] = "auto_repair_script"
                    stats["failed_repaired_to_success"] += 1
                else:
                    rec["status"] = "failed"
                    rec["repair_source"] = "auto_repair_partial"
                    issues = rec.get("description_retrieval_issues") or []
                    if "missing_core_metadata" not in issues:
                        issues = [*issues, "missing_core_metadata"]
                    rec["description_retrieval_issues"] = issues
                    rec["needs_regen"] = True
                    stats["failed_unresolved"] += 1

            if rec.get("synset") is None:
                stats["synset_missing_after"] += 1

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "input": str(args.input_jsonl),
        "output": str(args.output_jsonl),
        **{k: int(v) for k, v in stats.items()},
    }
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
