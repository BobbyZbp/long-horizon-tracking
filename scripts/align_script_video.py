#!/usr/bin/env python
"""
align_script_video.py

CLI entry point for the coarse script-to-video alignment layer.

This script consumes the narrative candidate layer produced before any
video-facing processing and
aligns candidate scene-chain steps to approximate subtitle-backed video time
windows.
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
