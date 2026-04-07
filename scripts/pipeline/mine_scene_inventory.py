#!/usr/bin/env python
"""
mine_scene_inventory.py

Build a high-recall per-scene object inventory from the screenplay.

What this file does
-------------------
- parses screenplay scenes
- extracts object evidence from action text and optionally dialogue
- merges both channels into a loose scene inventory with aliases and source
  channels preserved

Inputs
------
- `--pdf`: screenplay PDF
- `--out`: destination JSON path
- optional LLM/cache configuration and dialogue extraction mode

Outputs
-------
- writes one scene inventory JSON file, typically `scene_inventory.json`
- each scene contains `action_objects`, `dialogue_objects`, and
  `merged_loose_objects`

Workflow position
-----------------
- upstream: structural scene parsing and LLM extraction
- current stage: high-recall scene inventory
- downstream: climax scoring, Chekhov candidate mining, and later alignment
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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
