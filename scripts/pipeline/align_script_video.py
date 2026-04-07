#!/usr/bin/env python
"""
align_script_video.py

Build coarse subtitle-backed video windows for narrative candidates.

What this file does
-------------------
- loads validated narrative candidates, scene inventory, and subtitle timing
- aligns each supported scene step to an approximate movie interval
- records matched subtitle evidence plus fallback windows when dialogue overlap
  is weak

Inputs
------
- `--candidates`: `chekhov_candidates.json`
- `--scene_inventory`: `scene_inventory.json`
- `--subtitles`: `.srt`, `.vtt`, or normalized subtitle `.json`
- alignment hyperparameters such as search radius and default fallback window

Outputs
-------
- writes one alignment JSON file, typically `aligned_candidates.json`
- output records include aligned start/end times, matched subtitle text,
  confidence, and component scores

Workflow position
-----------------
- upstream: scene inventory + climax + candidate validation
- current stage: coarse script-to-video alignment
- downstream: candidate segment filtering and later frame/clip generation
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
from tracking_project.io.subtitles import load_subtitle_segments
from tracking_project.pipeline.video_alignment import build_script_video_alignment


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build coarse script-to-video alignment windows from subtitle timestamps.",
    )
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--scene_inventory", required=True)
    ap.add_argument("--subtitles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--search_radius_sec", type=float, default=420.0)
    ap.add_argument("--default_window_sec", type=float, default=75.0)
    ap.add_argument("--neighborhood", type=int, default=1)
    args = ap.parse_args()

    out_obj = build_script_video_alignment(
        load_json(args.candidates),
        load_json(args.scene_inventory),
        load_subtitle_segments(args.subtitles),
        search_radius_sec=args.search_radius_sec,
        default_window_sec=args.default_window_sec,
        neighborhood=args.neighborhood,
    )
    out_obj["meta"]["candidates_source"] = args.candidates
    out_obj["meta"]["scene_inventory_source"] = args.scene_inventory
    out_obj["meta"]["subtitles_source"] = args.subtitles

    safe_write_json(args.out, out_obj)
    print(
        f"[ok] aligned_windows={len(out_obj['alignments'])} -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
