# HSSD Standalone Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone, slot-aware HSSD asset retrieval pipeline from `hssd_asset_descriptions.jsonl` without touching existing scenesmith retrieval runtime.

**Architecture:** Normalize raw JSONL into structured `asset_index.jsonl` with deterministic category/support/composite fields, then run slot-gated candidate search with multi-factor rerank (category, semantic token overlap surrogate, style/support/color matches, negative penalties, composite penalties). Keep it decoupled from scenesmith service code.

**Tech Stack:** Python 3.11, JSONL, YAML configs, standard library only.

---

### Task 1: Scaffold standalone retrieval package

**Files:**
- Create: `asset_retrieval/asset_retrieval_core/__init__.py`
- Create: `asset_retrieval/asset_retrieval_core/normalization.py`
- Create: `asset_retrieval/asset_retrieval_core/retrieval.py`
- Create: `asset_retrieval/configs/*.yaml`
- Create: `asset_retrieval/scripts/01_normalize_assets.py`
- Create: `asset_retrieval/scripts/05_retrieve_for_scene.py`

- [ ] Define data model and config-driven normalization helpers.
- [ ] Implement standalone CLI for normalization.
- [ ] Implement standalone CLI for slot retrieval + rerank.

### Task 2: Implement robust normalization

**Files:**
- Modify: `asset_retrieval/asset_retrieval_core/normalization.py`
- Modify: `asset_retrieval/scripts/01_normalize_assets.py`

- [ ] Add synset-first category mapping with regex fallback.
- [ ] Add support class inference (`floor|wall|ceiling|surface|unknown`).
- [ ] Add retrieval text cleaning and composite detection.
- [ ] Emit quality flags and summary metrics.

### Task 3: Implement slot-aware retrieval and reranking

**Files:**
- Modify: `asset_retrieval/asset_retrieval_core/retrieval.py`
- Modify: `asset_retrieval/scripts/05_retrieve_for_scene.py`

- [ ] Enforce allowed category gating per slot.
- [ ] Apply negative terms and composite penalties.
- [ ] Add scoring function with explainable breakdown.
- [ ] Add pair binding mode (`same_asset`).

### Task 4: Add tests and verify with real data

**Files:**
- Create: `tests/unit/test_asset_retrieval_standalone.py`

- [ ] Add unit tests for category normalization/support/composite and rerank behavior.
- [ ] Run targeted tests.
- [ ] Run normalization script over full JSONL and verify counts.

