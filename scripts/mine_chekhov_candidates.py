#!/usr/bin/env python
"""
mine_chekhov_candidates.py

CLI entry point for narrative candidate mining before any alignment work.

This script should be run before any subtitle-based alignment or frame
sampling work.
It combines:
    - high-recall double-channel scene inventory
    - scene-level climax summaries
    - Stage-1 re-occurrence events as an optional strong prior

and exports a smaller set of "possible Chekhov" candidates with loose scene
chains plus cross-scene narrative validation reasons.
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
from tracking_project.pipeline.chekhov_candidates import build_chekhov_candidates


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build loose scene-level candidates and validate likely Chekhov patterns.",
    )
    ap.add_argument("--events", required=True)
    ap.add_argument("--scene_inventory", required=True)
    ap.add_argument("--scene_climax", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_validation_score", type=float, default=0.55)
    ap.add_argument("--top_k", type=int, default=0)
    args = ap.parse_args()

    out_obj = build_chekhov_candidates(
        load_json(args.events),
        load_json(args.scene_inventory),
        load_json(args.scene_climax),
        min_validation_score=args.min_validation_score,
        top_k=(args.top_k if args.top_k > 0 else None),
    )
    out_obj["meta"]["events_source"] = args.events
    out_obj["meta"]["scene_inventory_source"] = args.scene_inventory
    out_obj["meta"]["scene_climax_source"] = args.scene_climax

    safe_write_json(args.out, out_obj)
    print(
        f"[ok] possible_chekhov_candidates={len(out_obj['candidates'])} -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
