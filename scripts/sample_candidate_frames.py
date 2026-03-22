#!/usr/bin/env python
"""
sample_candidate_frames.py

CLI entry point for the frame sampling layer.

Pipeline position
-----------------
- prior layer: candidate segment filtering
- current layer: frame sampling
- next layer: SAM-based tracking demo

This script converts candidate segment timestamps into concrete frame files
copied into per-segment folders, plus a manifest that records the exact
timestamp-to-frame mapping used.
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from tracking_project.io.jsonio import load_json, safe_write_json
from tracking_project.pipeline.frame_sampling import sample_candidate_frames


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sample representative frames from candidate video segments.",
    )
    ap.add_argument("--segments", required=True)
    ap.add_argument("--frame_dir", required=True)
    ap.add_argument("--fps", required=True, type=float)
    ap.add_argument("--out_manifest", required=True)
    ap.add_argument("--out_frames_dir", required=True)
    ap.add_argument("--frame_index_base", type=int, default=0)
    ap.add_argument("--frames_per_segment", type=int, default=6)
    ap.add_argument(
        "--strategy",
        choices=["uniform", "first_middle_last"],
        default="uniform",
    )
    args = ap.parse_args()

    out_obj = sample_candidate_frames(
        load_json(args.segments),
        frame_dir=args.frame_dir,
        fps=args.fps,
        out_frames_dir=args.out_frames_dir,
        frame_index_base=args.frame_index_base,
        frames_per_segment=args.frames_per_segment,
        strategy=args.strategy,
    )
    out_obj["meta"]["segments_source"] = args.segments

    safe_write_json(args.out_manifest, out_obj)
    print(
        f"[ok] sampled_segments={len(out_obj['segments'])} -> {args.out_manifest}",
        flush=True,
    )


if __name__ == "__main__":
    main()
