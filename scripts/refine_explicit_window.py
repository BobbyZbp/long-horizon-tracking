#!/usr/bin/env python
"""
refine_explicit_window.py

CLI entry point for manual fixed-window refinement exports.

Why this file exists
--------------------
After a manual confirms the approximate visual interval for an object, we need a
small tool to export:
    - sparse review frames
    - contact sheets
    - a continuous mp4 clip
    - a continuous frame directory

This script exists for that manual handoff step before running a SAM3 notebook.
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from tracking_project.io.jsonio import safe_write_json
from tracking_project.pipeline.deliberate_resampling import refine_explicit_window


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export review and notebook-ready assets for one explicit time window.",
    )
    ap.add_argument("--label", required=True, help="Human label for the window, e.g. 'clip a'")
    ap.add_argument("--video", required=True, help="Source movie file")
    ap.add_argument("--out_dir", required=True, help="Output directory for this refined clip")
    ap.add_argument("--start_sec", type=float, required=True)
    ap.add_argument("--end_sec", type=float, required=True)
    ap.add_argument("--review_sample_every_sec", type=float, default=0.5)
    args = ap.parse_args()

    out_obj = refine_explicit_window(
        label=args.label,
        video_path=args.video,
        out_dir=args.out_dir,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
        review_sample_every_sec=args.review_sample_every_sec,
    )
    manifest_path = os.path.join(args.out_dir, "refinement_manifest.json")
    safe_write_json(manifest_path, out_obj)
    print(
        f"[ok] label={args.label} review_frames={out_obj['meta']['review_frames_returned']} "
        f"continuous_frames={out_obj['continuous_frames']['frames_written']} -> {manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
