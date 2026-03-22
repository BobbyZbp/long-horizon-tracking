# Coarse Alignment Layer Report

## Scope

This document defines the completed coarse alignment coding layer:

- subtitle input normalization
- coarse script-to-video alignment
- CLI interface
- output contract
- dependency on prior and next pipeline layers

This layer is complete at the code-and-test level.
Real-data execution still depends on external subtitle/video assets.

## Pipeline Placement

This alignment layer sits between:

- prior layer:
  - `chekhov_candidates.json`
  - `scene_inventory.json`
- current layer:
  - subtitle-backed coarse alignment
- next layer:
  - candidate segment filtering

The goal of this layer is not frame-accurate localization.
The goal is to produce believable, debuggable time windows for each narrative
scene step in a candidate's scene chain.

## Files

### 1. `src/tracking_project/io/subtitles.py`

Purpose:
- normalize subtitle-like sources into a deterministic time-ordered segment list

Why this file exists:
- alignment should not depend on ad hoc parsing logic inside a script
- the pipeline needs one stable movie-text timeline format

Accepted inputs:
- `.srt`
- `.vtt`
- normalized subtitle `.json`

Primary API:
- `load_subtitle_segments(path: str) -> List[SubtitleSegment]`

Output type:
- `SubtitleSegment(index, start_sec, end_sec, text)`

Required fields for normalized JSON:
- `segments`
- `segments[i].start_sec`
- `segments[i].end_sec`
- `segments[i].text`

Optional normalized JSON fields:
- `segments[i].index`

Prior layer:
- raw subtitle file or subtitle JSON

Next layer:
- `video_alignment.py`

### 2. `src/tracking_project/pipeline/video_alignment.py`

Purpose:
- convert narrative candidate scene-chain steps into approximate subtitle-backed
  video windows

Why this file exists:
- later tasks need time windows, not just scene indices
- candidate segment filtering and frame sampling cannot run directly on screenplay scene ids

Primary API:
- `build_script_video_alignment(candidates_json, scene_inventory_json, subtitle_segments, search_radius_sec=420.0, default_window_sec=75.0, neighborhood=1)`

Inputs:
- `candidates_json`
  - must be output from `chekhov_candidates.json`
- `scene_inventory_json`
  - used for `meta.scenes_non_omitted`
- `subtitle_segments`
  - output from `load_subtitle_segments(...)`
- `search_radius_sec`
  - maximum allowed distance from expected scene time when searching windows
- `default_window_sec`
  - fallback window size when no subtitle evidence is found
- `neighborhood`
  - how many subtitle neighbors to include when building each local text window

Output:
- JSON object with:
  - `meta`
  - `alignments`

Per-alignment fields:
- `candidate_index`
- `object_id`
- `canonical_name`
- `candidate_source`
- `scene_idx_non_omitted`
- `heading`
- `aligned_start_sec`
- `aligned_end_sec`
- `expected_time_sec`
- `confidence`
- `matched_subtitle_indices`
- `matched_subtitle_text`
- `query_terms`
- `component_scores.order_score`
- `component_scores.lexical_overlap`
- `component_scores.object_overlap`

Prior layer:
- `subtitles.py`
- `chekhov_candidates.py`
- `scene_inventory.py`

Next layer:
- `candidate_segments.py`

### 3. `scripts/align_script_video.py`

Purpose:
- CLI wrapper for the coarse alignment layer

Why this file exists:
- the alignment module should be runnable without importing Python manually
- a CLI is needed for reproducible experiments and later batch runs

Arguments:
- `--candidates`
- `--scene_inventory`
- `--subtitles`
- `--out`
- `--search_radius_sec`
- `--default_window_sec`
- `--neighborhood`

Output:
- JSON written with `safe_write_json(...)`

## Alignment Method

### Step 1: Expected movie time from story order

For a scene with non-omitted scene index `s`:

- `expected_time_sec = (s / (total_scenes - 1)) * movie_duration_sec`

This provides a coarse story-order prior.

### Step 2: Local subtitle windows

Each subtitle segment becomes a local search window.
The text of nearby subtitle segments can be merged into the window using the
`neighborhood` parameter.

### Step 3: Query terms from narrative evidence

Each scene step uses:
- candidate canonical name
- scene heading
- matched inventory object names
- scene inventory evidence snippets

These become:
- `general_terms`
- `object_terms`

### Step 4: Score each subtitle window

Current weighted score:

- `0.55 * order_score`
- `0.30 * lexical_overlap`
- `0.15 * object_overlap`

Rationale:
- order should dominate in the first alignment pass
- lexical/object overlap should refine, not fully replace, narrative ordering

### Step 5: Preserve scene-chain order

Within each candidate chain, final chosen windows are constrained to be
non-decreasing in time whenever possible.

Rationale:
- coarse alignment should not scramble story order

### Step 6: Fallback behavior

If no subtitle window is found within the search radius:
- use an expected-time-centered fallback window
- set `confidence = 0.0`
- leave subtitle evidence empty

Rationale:
- later tasks still need a deterministic interval
- no-match cases are meaningful debug outputs, not exceptions

## Data Contract

Canonical coarse-alignment output target:
- `data/aligned_segments/aligned_candidates.json`

Minimum required fields for candidate segment filtering:
- `object_id`
- `candidate_index`
- `scene_idx_non_omitted`
- `aligned_start_sec`
- `aligned_end_sec`
- `confidence`

Useful but optional downstream fields:
- `matched_subtitle_text`
- `matched_subtitle_indices`
- `query_terms`
- `component_scores`

## Tests

Alignment-layer tests:
- `tests/test_subtitles.py`
- `tests/test_video_alignment.py`

What is verified:
- subtitle timecode parsing
- normalized JSON subtitle loading
- alignment order preservation
- lexical/object overlap use
- fallback window behavior

## Current Status

Coding status:
- complete

Test status:
- passing

Real-run status:
- blocked on real subtitle input and movie/frame assets
