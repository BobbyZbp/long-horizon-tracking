#!/usr/bin/env python
"""
refine_windows.py

Unified CLI for the object-centric refinement layer.

What this file does
-------------------
- exposes two refinement modes from one entrypoint
- `from-object`: start from an existing candidate segment, expand around its
  center, and sample more densely for manual review
- `from-window`: start from a manually confirmed time interval and export
  review frames, contact sheets, a short clip, and continuous frames

Inputs
------
- `from-object` expects `candidate_segments.json`, an `object_id`, a source
  video path, and refinement parameters such as window radius / sample density
- `from-window` expects a manual label, a source video path, explicit start/end
  seconds, and review sampling density

Outputs
-------
- `from-object` writes a dense review manifest, usually
  `deliberate_resampling_manifest.json`, plus dense review frames and contact
  sheets
- `from-window` writes `refinement_manifest.json`, sparse review frames,
  contact sheets, a short mp4 clip, and a continuous frame directory

Workflow position
-----------------
- upstream: candidate segment filtering and human review
- current stage: object-centric manual refinement
- downstream: notebook-ready short clips and frame directories for SAM3-style
  visual validation
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
from tracking_project.pipeline.deliberate_resampling import (
    deliberate_resample_object,
    refine_explicit_window,
)


def _run_from_object(args: argparse.Namespace) -> None:
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
    manifest_path = Path(args.out_dir) / "deliberate_resampling_manifest.json"
    safe_write_json(str(manifest_path), out_obj)
    print(
        f"[ok] object={args.object_id} frames={out_obj['meta']['frames_returned']} -> {manifest_path}",
        flush=True,
    )


def _run_from_window(args: argparse.Namespace) -> None:
    out_obj = refine_explicit_window(
        label=args.label,
        video_path=args.video,
        out_dir=args.out_dir,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
        review_sample_every_sec=args.review_sample_every_sec,
    )
    manifest_path = Path(args.out_dir) / "refinement_manifest.json"
    safe_write_json(str(manifest_path), out_obj)
    print(
        f"[ok] label={args.label} review_frames={out_obj['meta']['review_frames_returned']} "
        f"continuous_frames={out_obj['continuous_frames']['frames_written']} -> {manifest_path}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Object-centric refinement entrypoint for dense review exports and explicit windows.",
    )
    sub = ap.add_subparsers(dest="mode", required=True)

    from_object = sub.add_parser(
        "from-object",
        help="Create a dense centered review set for one object segment.",
    )
    from_object.add_argument("--segments", required=True, help="Path to candidate_segments.json")
    from_object.add_argument("--object_id", required=True, help="Target object_id")
    from_object.add_argument("--video", required=True, help="Source movie file")
    from_object.add_argument("--out_dir", required=True, help="Root directory for outputs")
    from_object.add_argument("--segment_id", default=None, help="Optional explicit segment id")
    from_object.add_argument(
        "--center_sec",
        type=float,
        default=None,
        help="Optional manual center timestamp in seconds for the review window",
    )
    from_object.add_argument("--window_radius_sec", type=float, default=120.0)
    from_object.add_argument("--sample_every_sec", type=float, default=1.0)
    from_object.set_defaults(handler=_run_from_object)

    from_window = sub.add_parser(
        "from-window",
        help="Export review and notebook-ready assets for one explicit time window.",
    )
    from_window.add_argument("--label", required=True, help="Human label for the window, e.g. 'clip a'")
    from_window.add_argument("--video", required=True, help="Source movie file")
    from_window.add_argument("--out_dir", required=True, help="Output directory for this refined clip")
    from_window.add_argument("--start_sec", type=float, required=True)
    from_window.add_argument("--end_sec", type=float, required=True)
    from_window.add_argument("--review_sample_every_sec", type=float, default=0.5)
    from_window.set_defaults(handler=_run_from_window)

    args = ap.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
