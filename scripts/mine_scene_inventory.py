#!/usr/bin/env python
"""
mine_scene_inventory.py

CLI entry point for building the high-recall double-channel scene inventory
required before cross-scene narrative validation and any video-facing
alignment.

This script should be run before loose candidate mining and before any
script-to-video alignment work. It produces:
    - action_objects
    - dialogue_objects
    - merged_loose_objects

for every non-omitted scene in the screenplay.
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
from tracking_project.pipeline.scene_inventory import build_scene_inventory


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a high-recall double-channel scene inventory.",
    )
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--llm_model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    ap.add_argument("--cache_dir", default="data/llm_cache")
    ap.add_argument(
        "--dialogue_mode",
        choices=["llm", "heuristic", "hybrid", "none"],
        default="hybrid",
    )
    args = ap.parse_args()

    out_obj = build_scene_inventory(
        pdf=args.pdf,
        llm_model=args.llm_model,
        cache_dir=args.cache_dir,
        dialogue_mode=args.dialogue_mode,
    )
    safe_write_json(args.out, out_obj)
    print(
        f"[ok] scene_inventory_scenes={len(out_obj['scenes'])} -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
