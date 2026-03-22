# Narrative Preparation Pipeline Spec

## Purpose

This document defines the exact pre-alignment pipeline state that must exist
before coarse script-to-video alignment begins. It formalizes the current
narrative-layer architecture, the files we added, and the contracts between
layers.

This stage in the project is not a model-training step. It is a
pipeline-definition step. Its deliverable is a fixed, explicit data flow with
concrete file interfaces.

## Current Pipeline Order

The correct order before any video-side localization is:

1. screenplay PDF
2. scene-level high-recall object inventory
3. scene-level climax summaries
4. loose Chekhov candidate mining
5. cross-scene narrative validation
6. coarse script-to-video alignment

This ordering is intentional.

We do not start from Stage 1 alone because Stage 1 is already a filter.
We do not start from tracking because tracking needs a candidate object list
and approximate video windows first.

## Layer Definitions

### Layer A: Structural Scene Parsing

File:
- `src/tracking_project/io/pdf_scene_parser.py`

Purpose:
- Parse the screenplay PDF into ordered `Scene` objects.

Input:
- `pdf_path: str`

Output:
- `Tuple[List[Scene], Dict]`

Important output fields:
- `Scene.scene_idx`
- `Scene.scene_idx_non_omitted`
- `Scene.heading`
- `Scene.page_start`
- `Scene.page_end`
- `Scene.lines`

Argument requirements:
- `pdf_path` must point to the screenplay PDF.
- Real PDF parsing requires `pdfplumber`, unless a custom `pdf_open` is passed
  in tests.

Why this layer exists:
- Every later layer depends on stable scene numbering.
- Scene indices are the bridge between narrative reasoning and later video
  alignment.

### Layer B: High-Recall Scene Inventory

Files:
- `src/tracking_project/pipeline/scene_inventory.py`
- `scripts/mine_scene_inventory.py`

Purpose:
- Build a per-scene object inventory with higher recall than Stage 1.
- Preserve both action-based and dialogue-based object evidence.

Primary function:
- `build_scene_inventory(pdf, llm_model="gpt-4o-mini", cache_dir="data/llm_cache", keep_categories=None, dialogue_mode="hybrid")`

Inputs:
- `pdf: str`
- `llm_model: str`
- `cache_dir: str`
- `keep_categories: Sequence[str] | None`
- `dialogue_mode: str` in `{"llm", "heuristic", "hybrid", "none"}`

Output:
- JSON object with:
  - `meta`
  - `scenes`

Per-scene output fields:
- `heading`
- `page_start`
- `page_end`
- `action_objects`
- `dialogue_objects`
- `merged_loose_objects`

`action_objects` record fields:
- `object_id`
- `canonical_name`
- `surface_forms`
- `category`
- `physical_visualizable`
- `direct_visual_evidence`
- `mention_strength`
- `confidence`
- `evidence`
- `source_channel`
- `tracking_keep`

`dialogue_objects` record fields:
- same core fields as above, except dialogue support does not depend on Stage 1
  `keep`

`merged_loose_objects` record fields:
- `object_id`
- `canonical_name`
- `surface_forms`
- `source_channels`
- `normalized_aliases`
- `category`
- `physical_visualizable`
- `direct_visual_evidence`
- `mention_strength_max`
- `confidence_max`
- `evidence`
- `tracking_keep_any`

Why this layer exists:
- Stage 1 is precision-oriented and can miss setup objects.
- Chekhov-style reasoning needs recall first, then validation.
- Dialogue mentions matter for setup objects, even if they are not yet strong
  tracking targets.

Why `merged_loose_objects` exists:
- Cross-scene candidate mining should not have to separately reconcile action
  and dialogue mentions for every scene.
- The merged layer preserves source channels so downstream code can still tell
  the difference between strong action evidence and weaker dialogue-only
  evidence.

CLI:
- `python scripts/mine_scene_inventory.py --pdf <pdf> --out <json> [--llm_model ...] [--cache_dir ...] [--dialogue_mode ...]`

Canonical output file:
- `data/events/scene_inventory.json`

### Layer C: Scene Climax Summaries

Existing files:
- `src/tracking_project/pipeline/climax_mine.py`
- `scripts/mine_climax_llm.py`

Purpose:
- Provide per-scene climax scores.

Current output file already present in repo:
- `data/events/scene_climax.json`

Minimum fields required by downstream layers:
- `scenes.<scene_idx>.heading`
- `scenes.<scene_idx>.climax_score`

Why this layer still matters:
- Chekhov payoff scenes should preferentially land in later, higher-climax
  narrative regions.
- The coarse alignment layer can use these later scores as a relevance prior.

### Layer D: Loose Candidate Mining + Cross-Scene Narrative Validation

Files:
- `src/tracking_project/pipeline/chekhov_candidates.py`
- `scripts/mine_chekhov_candidates.py`

Purpose:
- Aggregate object evidence across scenes.
- Use Stage 1 as a strong prior, not a hard gate.
- Produce a smaller set of narrative candidates safe to pass into alignment.

Primary function:
- `build_chekhov_candidates(events_json, scene_inventory_json, scene_climax_json, min_validation_score=0.55, top_k=None)`

Inputs:
- `events_json: Dict[str, Any]`
- `scene_inventory_json: Dict[str, Any]`
- `scene_climax_json: Dict[str, Any]`
- `min_validation_score: float`
- `top_k: int | None`

Output:
- JSON object with:
  - `meta`
  - `field_spec`
  - `candidates`

Candidate record fields:
- `object_id`
- `canonical_name`
- `entity_type`
- `candidate_source`
- `candidate_stats`
- `inventory_support`
- `stage1_prior`
- `possible_chekhov_score`
- `possible_chekhov_label`
- `validation_reasons`
- `setup_scene_indices`
- `payoff_scene_indices`
- `loose_scene_chain`

`candidate_source` values:
- `inventory_with_stage1_prior`
- `inventory_only`
- `stage1_fallback`

Why this layer exists:
- High-recall inventory alone is too broad.
- Stage 1 alone is too narrow.
- This layer combines both to produce narratively plausible Chekhov candidates.

CLI:
- `python scripts/mine_chekhov_candidates.py --events <run1.json> --scene_inventory <scene_inventory.json> --scene_climax <scene_climax.json> --out <json> [--min_validation_score ...] [--top_k ...]`

Canonical output file:
- `data/events/chekhov_candidates.json`

## Formal Input/Output Chain Before Coarse Alignment

### Canonical inputs

- `data/raw/harry-potter-and-the-chamber-of-secrets-2002.pdf`
- `data/events/run1.json`
- `data/events/scene_climax.json`
- `data/events/scene_inventory.json`

### Canonical intermediate file

- `data/events/chekhov_candidates.json`

### Canonical upstream object for Coarse Alignment

The coarse alignment layer must start from:
- `data/events/chekhov_candidates.json`

The coarse alignment layer should not start directly from:
- `run1.json`
- `scene_climax.json`

Reason:
- `chekhov_candidates.json` is the first file where object-level candidate
  status, cross-scene support, setup/payoff scenes, and narrative reasons are
  already combined into one stable structure.

## One Concrete Example Contract

One candidate should be traceable as:

1. `scene_inventory.json`
   - multiple scenes contain `merged_loose_objects` matching the same object
2. `scene_climax.json`
   - those scenes have climax scores
3. `run1.json`
   - optional Stage 1 prior confirms a large reappearance gap
4. `chekhov_candidates.json`
   - final candidate record includes:
     - candidate object id
     - setup scenes
     - payoff scenes
     - loose scene chain
     - validation score

## Readiness State

What is complete:
- High-recall scene inventory code
- Candidate mining code
- Unit tests for inventory and candidate validation
- PDF parser smoke-test guard for environments without `pdfplumber`

What is not yet complete in this environment:
- Real regeneration of `scene_inventory.json`
- Real regeneration of `chekhov_candidates.json`

Reason:
- Current local Python interpreter does not have `pdfplumber` or `openai`

This is an environment limitation, not a pipeline-design limitation.

## Decision

This narrative-preparation stage is considered complete when:
- this interface definition is fixed
- the narrative-layer code and tests pass
- the coarse alignment layer uses `chekhov_candidates.json` as its narrative upstream input
