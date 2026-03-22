# Candidate Segment Filtering and Frame Sampling Report

## Scope

This document defines the completed coding layers for:

- candidate segment filtering
- frame sampling

Both layers are complete at the code-and-test level.
Both layers still require real input assets to produce final experiment outputs.

## Pipeline Placement

Candidate segment filtering sits between:
- prior layer: coarse alignment
- current layer: candidate segment filtering
- next layer: frame sampling

Frame sampling sits between:
- prior layer: candidate segment filtering
- current layer: frame sampling
- next layer: SAM-based tracking demo

## Candidate Segment Filtering Layer

### File: `src/tracking_project/pipeline/candidate_segments.py`

Purpose:
- reduce many coarse aligned scene windows into a smaller set of object-level
  candidate video segments

Why this file exists:
- The coarse alignment layer is scene-step level and intentionally coarse
- those windows are too numerous and too fragmented for direct tracking
- the project needs a controlled reduction step before touching frames

Primary API:
- `build_candidate_segments(candidates_json, alignment_json, min_alignment_confidence=0.15, padding_sec=12.0, merge_gap_sec=8.0, max_segments_per_object=4)`

Inputs:
- `candidates_json`
  - output from `chekhov_candidates.json`
- `alignment_json`
  - output from the coarse alignment layer
- `min_alignment_confidence`
  - seed threshold for keeping an aligned scene step
- `padding_sec`
  - expansion on both sides of each aligned window
- `merge_gap_sec`
  - maximum temporal gap for merging nearby windows
- `max_segments_per_object`
  - cap after ranking

Output:
- JSON object with:
  - `meta`
  - `segments`

Per-segment fields:
- `segment_id`
- `candidate_index`
- `object_id`
- `canonical_name`
- `candidate_source`
- `possible_chekhov_score`
- `source_scene_indices`
- `source_alignment_indices`
- `segment_role`
- `segment_start_sec`
- `segment_end_sec`
- `priority_score`
- `alignment_confidence_max`
- `reason_tags`
- `matched_subtitle_texts`
- `matched_subtitle_indices`
- `query_terms`

How candidate segment filtering works:

1. seed from coarse alignment windows
2. attach scene role tags from the narrative candidate chain
3. expand aligned windows by `padding_sec`
4. score each seed segment using:
   - candidate Chekhov score
   - alignment confidence
   - scene role bonuses
5. merge nearby windows for the same object
6. keep only the top segments per object

Why `segment_role` exists:
- downstream tasks need to know whether a segment is closer to setup, payoff,
  reappearance, or just a bridge segment

Why `reason_tags` exist:
- later debugging should not rely on hidden heuristics
- every segment should explain why it survived this layer

### File: `scripts/filter_candidate_segments.py`

Purpose:
- CLI wrapper for candidate segment filtering

Arguments:
- `--candidates`
- `--alignments`
- `--out`
- `--min_alignment_confidence`
- `--padding_sec`
- `--merge_gap_sec`
- `--max_segments_per_object`

Output target:
- `data/candidate_objects/candidate_segments.json`

## Frame Sampling Layer

### File: `src/tracking_project/pipeline/frame_sampling.py`

Purpose:
- convert candidate segments into real frame files plus a manifest that records
  the exact timestamp-to-frame mapping

Why this file exists:
- Candidate segment filtering only returns time windows
- Downstream tracking cannot use time windows directly
- the project needs a deterministic visual inspection layer before SAM

Primary API:
- `sample_candidate_frames(segments_json, frame_dir, fps, out_frames_dir, frame_index_base=0, frames_per_segment=6, strategy="uniform")`

Inputs:
- `segments_json`
  - output from candidate segment filtering
- `frame_dir`
  - directory containing extracted movie frames
- `fps`
  - fps of the extraction source
- `out_frames_dir`
  - where copied sampled frames should be written
- `frame_index_base`
  - whether frame filenames start from 0 or 1
- `frames_per_segment`
  - number of representative frames to copy
- `strategy`
  - one of:
    - `uniform`
    - `first_middle_last`

Output:
- JSON object with:
  - `meta`
  - `segments`

Per-segment output:
- `segment_id`
- `object_id`
- `canonical_name`
- `segment_start_sec`
- `segment_end_sec`
- `fps`
- `frame_index_base`
- `sampling_strategy`
- `sampled_frames`

Per-sampled-frame output:
- `sample_index`
- `timestamp_sec`
- `target_frame_index`
- `frame_index`
- `source_path`
- `output_path`

How frame sampling works:

1. index the extracted frame directory
2. choose representative timestamps for each segment
3. convert each timestamp to a target frame index
4. snap to the nearest available real frame if necessary
5. copy sampled frames into a segment-specific output folder
6. write a manifest that records the exact mapping

Why nearest-frame snapping exists:
- extracted frame folders are not always perfect or complete
- the pipeline should degrade gracefully instead of failing on one missing image

### File: `scripts/sample_candidate_frames.py`

Purpose:
- CLI wrapper for frame sampling

Arguments:
- `--segments`
- `--frame_dir`
- `--fps`
- `--out_manifest`
- `--out_frames_dir`
- `--frame_index_base`
- `--frames_per_segment`
- `--strategy`

Output targets:
- manifest:
  - `data/sampled_frames/sampled_frames_manifest.json`
- copied frame folders:
  - `data/sampled_frames/<segment_id>/...`

## Tests

### Candidate segment filtering tests
- `tests/test_candidate_segments.py`

Verified behavior:
- nearby aligned windows merge into one segment
- reason tags are preserved
- segment cap per object works

### Frame sampling tests
- `tests/test_frame_sampling.py`

Verified behavior:
- uniform frame sampling works
- output files are copied into segment folders
- nearest available frame fallback works when exact indices are missing

## Current Status

Candidate segment filtering status:
- complete

Frame sampling status:
- complete

Test status:
- passing

Real-run status:
- blocked on real alignment outputs, real frame directory, and fps metadata
