#!/usr/bin/env python
"""
filter_candidate_segments.py

Consolidate coarse aligned windows into object-level candidate segments.

What this file does
-------------------
- loads candidate objects plus coarse aligned windows
- expands aligned windows into broader visual search segments
- merges nearby windows that likely belong to the same object-level interval
- scores and caps the surviving segments per object

Inputs
------
- `--candidates`: `chekhov_candidates.json`
- `--alignments`: `aligned_candidates.json`
- filtering parameters such as confidence threshold, padding, merge gap, and
  per-object segment cap

Outputs
-------
- writes one candidate-segment JSON file, typically `candidate_segments.json`
- output records include segment time bounds, priority score, reason tags, and
  source scene/alignment provenance

Workflow position
-----------------
- upstream: coarse script-to-video alignment
- current stage: candidate segment filtering
- downstream: default frame sampling or manual object-centric refinement
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tracking_project.io.jsonio import load_json, safe_write_json
from tracking_project.pipeline.candidate_segments import build_candidate_segments


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build candidate video segments from coarse alignment windows.",
    )
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--alignments", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_alignment_confidence", type=float, default=0.15)
    ap.add_argument("--padding_sec", type=float, default=12.0)
    ap.add_argument("--merge_gap_sec", type=float, default=8.0)
    ap.add_argument("--max_segments_per_object", type=int, default=4)
    args = ap.parse_args()

    out_obj = build_candidate_segments(
        load_json(args.candidates),
        load_json(args.alignments),
        min_alignment_confidence=args.min_alignment_confidence,
        padding_sec=args.padding_sec,
        merge_gap_sec=args.merge_gap_sec,
        max_segments_per_object=args.max_segments_per_object,
    )
    out_obj["meta"]["candidates_source"] = args.candidates
    out_obj["meta"]["alignments_source"] = args.alignments

    safe_write_json(args.out, out_obj)
    print(
        f"[ok] candidate_segments={len(out_obj['segments'])} -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
