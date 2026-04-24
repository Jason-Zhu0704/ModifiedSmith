from __future__ import annotations

import json
import tempfile

from pathlib import Path

from asset_retrieval.asset_retrieval_core.normalization import (
    infer_support_class,
    load_normalization_config,
    normalize_asset,
    normalize_assets_stream,
)


def _config() -> dict:
    return load_normalization_config(Path("/data/scenesmith/asset_retrieval/configs"))


def test_surface_support_class_for_table_lamp_even_if_small_object() -> None:
    support = infer_support_class(
        {"category_buckets": ["small_objects"]}, slot_category="table_lamp"
    )
    assert support == "surface"


def test_synset_mapping_to_slot_category() -> None:
    asset = {
        "asset_id": "a1",
        "mesh_id": "a1",
        "name": "Petra Night Stand",
        "synset": "nightstand.n.01",
        "description": "A modern bedside table with two drawers.",
        "category_buckets": ["large_objects"],
        "status": "success",
        "needs_regen": False,
    }
    normalized = normalize_asset(asset, _config())
    assert normalized["slot_category"] == "nightstand"
    assert normalized["quality_flags"]["category_confidence"] == 0.95


def test_keyword_fallback_when_synset_missing() -> None:
    asset = {
        "asset_id": "a2",
        "mesh_id": "a2",
        "name": "Simple Desk",
        "synset": None,
        "description": "A writing desk with compact workspace.",
        "category_buckets": ["large_objects"],
        "status": "success",
        "needs_regen": False,
    }
    normalized = normalize_asset(asset, _config())
    assert normalized["slot_category"] == "desk"
    assert normalized["quality_flags"]["category_confidence"] == 0.70


def test_synset_token_fallback_when_not_in_direct_map() -> None:
    asset = {
        "asset_id": "a2b",
        "mesh_id": "a2b",
        "name": "Tea Light Holder",
        "synset": "candlestick.n.01",
        "description": "A small decorative holder.",
        "category_buckets": ["small_objects"],
        "status": "success",
        "needs_regen": False,
    }
    normalized = normalize_asset(asset, _config())
    assert normalized["slot_category"] == "vase"
    assert normalized["quality_flags"]["category_confidence"] == 0.55


def test_noise_phrase_removed_from_retrieval_clean_text() -> None:
    asset = {
        "asset_id": "a3",
        "mesh_id": "a3",
        "name": "Flower Pot",
        "synset": "flower_in_vase.n.01",
        "description": "A flower arrangement shown in various angles against a neutral gray background.",
        "category_buckets": ["small_objects"],
        "status": "success",
        "needs_regen": False,
    }
    normalized = normalize_asset(asset, _config())
    text = normalized["retrieval_clean_text"].lower()
    assert "against a neutral gray background" not in text
    assert "shown in various" not in text


def test_composite_asset_detection() -> None:
    asset = {
        "asset_id": "a4",
        "mesh_id": "a4",
        "name": "Modern dining setup",
        "synset": None,
        "description": "A dining setup with round table surrounded by six chairs.",
        "key_features": ["six chairs", "table"],
        "category_buckets": ["large_objects"],
        "status": "success",
        "needs_regen": False,
    }
    normalized = normalize_asset(asset, _config())
    assert normalized["is_composite"] is True


def test_normalize_assets_stream_writes_summary() -> None:
    cfg = _config()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        input_path = tmp / "input.jsonl"
        output_path = tmp / "output.jsonl"

        records = [
            {
                "asset_id": "a1",
                "mesh_id": "a1",
                "name": "Petra Night Stand",
                "synset": "nightstand.n.01",
                "description": "A modern bedside table with two drawers.",
                "category_buckets": ["large_objects"],
                "status": "success",
                "needs_regen": False,
            },
            {
                "asset_id": "a2",
                "mesh_id": "a2",
                "name": "Unknown Item",
                "synset": None,
                "description": "An ambiguous object.",
                "category_buckets": ["small_objects"],
                "status": "failed",
                "needs_regen": True,
            },
        ]

        input_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
            encoding="utf-8",
        )

        summary = normalize_assets_stream(input_path, output_path, cfg)

        assert summary["total"] == 2
        assert summary["success"] == 1
        assert summary["failed"] == 1
        assert output_path.exists()
        assert output_path.read_text(encoding="utf-8").count("\n") == 2
