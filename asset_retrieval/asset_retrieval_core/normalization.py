from __future__ import annotations

import json
import re

from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_normalization_config(config_dir: Path) -> dict[str, Any]:
    keyword_raw = _load_yaml(config_dir / "keyword_map.yaml")
    keyword_map: list[tuple[re.Pattern[str], str]] = []
    for item in keyword_raw:
        pattern = item["pattern"]
        category = item["category"]
        keyword_map.append((re.compile(pattern, re.IGNORECASE), category))

    return {
        "synset_map": _load_yaml(config_dir / "synset_map.yaml") or {},
        "keyword_map": keyword_map,
        "noise_phrases": _load_yaml(config_dir / "noise_phrases.yaml") or [],
        "composite_hints": _load_yaml(config_dir / "composite_hints.yaml") or [],
        "feature_whitelist": _load_yaml(config_dir / "feature_whitelist.yaml") or {},
    }


def normalize_category(asset: dict[str, Any], config: dict[str, Any]) -> tuple[str, float]:
    synset = asset.get("synset")
    synset_map: dict[str, str] = config["synset_map"]
    if synset in synset_map:
        return synset_map[synset], 0.95

    if isinstance(synset, str) and synset:
        synset_lemma = synset.split(".")[0].replace("_", " ").lower()
        synset_rules: list[tuple[str, str]] = [
            (r"\bnightstand\b|\bbedside\b", "nightstand"),
            (r"\bbed\b", "bed"),
            (r"\bdesk\b", "desk"),
            (r"\bchair\b", "chair"),
            (r"\bsofa\b", "sofa"),
            (r"\bwardrobe\b", "wardrobe"),
            (r"\bdresser\b", "dresser"),
            (r"\bbookcase\b|\bbookshelf\b|\bshelf\b", "bookcase"),
            (r"\bcabinet\b", "cabinet"),
            (r"\brug\b|\bcarpet\b|\bmat\b", "rug"),
            (r"\bcoffee table\b", "coffee_table"),
            (r"\bdining table\b", "dining_table"),
            (r"\btable\b", "table"),
            (r"\bfloor lamp\b", "floor_lamp"),
            (r"\btable lamp\b", "table_lamp"),
            (r"\bwall lamp\b", "wall_lamp"),
            (r"\bceiling lamp\b|\bpendant\b|\bchandelier\b", "ceiling_lamp"),
            (r"\bmirror\b", "wall_mirror"),
            (r"\bwall art\b|\bpainting\b|\bposter\b|\bprint\b", "wall_art"),
            (r"\bplant\b|\bflower\b", "plant"),
            (r"\bvase\b|\bcandlestick\b", "vase"),
            (r"\brefrigerator\b", "refrigerator"),
            (r"\boven\b", "oven"),
            (r"\bsink\b", "sink"),
            (r"\btoilet\b", "toilet"),
            (r"\bshower\b", "shower_fixture"),
        ]
        for pattern, category in synset_rules:
            if re.search(pattern, synset_lemma):
                return category, 0.55

    text = " ".join(
        [
            asset.get("name") or "",
            asset.get("description") or "",
            asset.get("description_retrieval") or "",
        ]
    ).lower()

    for pattern, category in config["keyword_map"]:
        if pattern.search(text):
            return category, 0.70

    return "unknown", 0.0


def infer_support_class(asset: dict[str, Any], slot_category: str) -> str:
    buckets = set(asset.get("category_buckets") or [])

    if "wall_objects" in buckets:
        return "wall"
    if "ceiling_objects" in buckets:
        return "ceiling"

    if slot_category in {
        "vase",
        "flower_in_vase",
        "table_lamp",
        "throw_pillow",
        "book",
        "laptop",
        "clock",
        "candlestick",
    }:
        return "surface"

    if "large_objects" in buckets or slot_category in {
        "bed",
        "desk",
        "chair",
        "office_chair",
        "nightstand",
        "wardrobe",
        "dresser",
        "rug",
        "sofa",
        "bench",
        "plant",
        "floor_lamp",
        "bookcase",
        "cabinet",
    }:
        return "floor"

    return "unknown"


def clean_text(text: str, noise_phrases: list[str]) -> str:
    out = text or ""
    for phrase in noise_phrases:
        out = re.sub(re.escape(phrase), " ", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def infer_is_composite(asset: dict[str, Any], composite_hints: list[str]) -> bool:
    raw_features = asset.get("key_features") or []
    features = [str(f) for f in raw_features]
    text = " ".join(
        [
            asset.get("name") or "",
            asset.get("description") or "",
            " ".join(features),
        ]
    ).lower()
    return any(h.lower() in text for h in composite_hints)


def filter_features_for_category(
    category: str, features: list[str], feature_whitelist: dict[str, list[str]]
) -> list[str]:
    normalized_features = [str(f) for f in features]
    allowed = feature_whitelist.get(category)
    if not allowed:
        return normalized_features[:4]

    out: list[str] = []
    lowered_allowed = [k.lower() for k in allowed]
    for feature in normalized_features:
        fl = feature.lower()
        if any(k in fl for k in lowered_allowed):
            out.append(feature)
    return out[:6]


def build_retrieval_clean_text(
    asset: dict[str, Any], slot_category: str, config: dict[str, Any]
) -> str:
    name = asset.get("name") or ""
    desc = clean_text(
        asset.get("description") or asset.get("description_retrieval") or "",
        config["noise_phrases"],
    )
    style = asset.get("style") or ""
    material = asset.get("primary_material") or ""
    color = asset.get("color_tone") or ""
    features = filter_features_for_category(
        slot_category,
        asset.get("key_features") or [],
        config["feature_whitelist"],
    )

    return (
        f"category: {slot_category}. "
        f"name: {name}. "
        f"style: {style}. "
        f"material: {material}. "
        f"color: {color}. "
        f"description: {desc}. "
        f"features: {', '.join(features)}."
    )


def normalize_asset(asset: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    slot_category, category_confidence = normalize_category(asset, config)
    support_class = infer_support_class(asset, slot_category)
    is_composite = infer_is_composite(asset, config["composite_hints"])

    return {
        "asset_id": asset.get("asset_id"),
        "mesh_id": asset.get("mesh_id"),
        "raw_name": asset.get("name"),
        "raw_synset": asset.get("synset"),
        "slot_category": slot_category,
        "support_class": support_class,
        "is_composite": is_composite,
        "is_scene_or_set": is_composite,
        "retrieval_clean_text": build_retrieval_clean_text(asset, slot_category, config),
        "style_tags": [asset.get("style")] if asset.get("style") else [],
        "material_tags": [asset.get("primary_material")]
        if asset.get("primary_material")
        else [],
        "color_tags": [asset.get("color_tone")] if asset.get("color_tone") else [],
        "category_buckets": asset.get("category_buckets") or [],
        "quality_flags": {
            "status": asset.get("status"),
            "needs_regen": bool(asset.get("needs_regen")),
            "synset_missing": asset.get("synset") is None,
            "category_confidence": category_confidence,
        },
        "raw": asset,
    }


def normalize_assets_stream(
    input_path: Path, output_path: Path, config: dict[str, Any]
) -> dict[str, Any]:
    total = 0
    success = 0
    failed = 0
    synset_missing = 0
    unknown_category = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for raw_line in fin:
            if not raw_line.strip():
                continue
            total += 1
            asset = json.loads(raw_line)

            normalized = normalize_asset(asset, config)
            fout.write(json.dumps(normalized, ensure_ascii=False) + "\n")

            status = normalized["quality_flags"]["status"]
            if status == "success":
                success += 1
            else:
                failed += 1

            if normalized["quality_flags"]["synset_missing"]:
                synset_missing += 1
            if normalized["slot_category"] == "unknown":
                unknown_category += 1

    return {
        "total": total,
        "success": success,
        "failed": failed,
        "synset_missing": synset_missing,
        "unknown_category": unknown_category,
        "output_path": str(output_path),
    }
