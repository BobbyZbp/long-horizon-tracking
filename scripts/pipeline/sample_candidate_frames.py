#!/usr/bin/env python
"""
sample_candidate_frames.py

Sample representative still frames from candidate video segments.

What this file does
-------------------
- loads candidate segments and an extracted frame directory
- converts segment timestamps into target frame indices
- copies representative frames into per-segment output folders
- records the exact timestamp-to-frame mapping in a manifest

Inputs
------
- `--segments`: `candidate_segments.json`
- `--frame_dir`: directory of pre-extracted movie frames
- `--fps`: fps used when frames were extracted
- `--out_manifest` and `--out_frames_dir`
- optional sampling strategy parameters

Outputs
-------
- writes a sampled-frame manifest JSON
- writes copied representative frame images under the requested output folder

Workflow position
-----------------
- upstream: candidate segment filtering
- current stage: default broad frame sampling for visual review
- downstream: manual inspection or a later tracking demo
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
