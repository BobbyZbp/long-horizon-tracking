#!/usr/bin/env python
"""
deliberate_resample_object.py

CLI entry point for manual-first, object-centric dense resampling.

Why this file exists
--------------------
The main frame-sampling script is designed for broad candidate review. This
script exists for a narrower workflow: take one chosen object, expand its best
current segment around the midpoint, sample densely, and save the result for
human review before preparing a short SAM3 clip.

Pipeline position
-----------------
- prior layer: candidate segment filtering
- current layer: deliberate object-centric resampling
- next layer: manual review and notebook-ready clip preparation
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
from tracking_project.pipeline.deliberate_resampling import deliberate_resample_object


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create a dense centered review set for one object segment.",
    )
    ap.add_argument("--segments", required=True, help="Path to candidate_segments.json")
    ap.add_argument("--object_id", required=True, help="Target object_id")
    ap.add_argument("--video", required=True, help="Source movie file")
    ap.add_argument("--out_dir", required=True, help="Root directory for outputs")
    ap.add_argument("--segment_id", default=None, help="Optional explicit segment id")
    ap.add_argument(
        "--center_sec",
        type=float,
        default=None,
        help="Optional manual center timestamp in seconds for the review window",
    )
    ap.add_argument("--window_radius_sec", type=float, default=120.0)
    ap.add_argument("--sample_every_sec", type=float, default=1.0)
    args = ap.parse_args()

    out_obj = deliberate_resample_object(
        load_json(args.segments),
        object_id=args.object_id,
        video_path=args.video,
        out_dir=args.out_dir,
        segment_id=args.segment_id,
        window_radius_sec=args.window_radius_sec,
        sample_every_sec=args.sample_every_sec,
        center_sec=args.center_sec,
    )
    out_obj["meta"]["segments_source"] = args.segments
    manifest_path = os.path.join(args.out_dir, "deliberate_resampling_manifest.json")
    safe_write_json(manifest_path, out_obj)
    print(
        f"[ok] object={args.object_id} frames={out_obj['meta']['frames_returned']} -> {manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
