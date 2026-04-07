#!/usr/bin/env python
"""
mine_chekhov_candidates.py

Build cross-scene narrative candidates before any video alignment work.

What this file does
-------------------
- combines scene-level inventory support, climax scores, and optional Stage-1
  long-gap priors
- validates which objects exhibit a plausible setup -> absence -> payoff
  pattern
- produces a smaller set of narratively supported object candidates for later
  alignment

Inputs
------
- `--events`: Stage-1 events JSON, usually `run1.json`
- `--scene_inventory`: `scene_inventory.json`
- `--scene_climax`: `scene_climax.json`
- candidate-selection parameters such as minimum validation score and optional
  top-k cap

Outputs
-------
- writes one candidate JSON file, typically `chekhov_candidates.json`
- output records include setup/payoff scenes, loose scene chains, support
  traces, and narrative validation reasons

Workflow position
-----------------
- upstream: Stage-1 priors, scene inventory, and climax summaries
- current stage: loose candidate mining + cross-scene narrative validation
- downstream: coarse script-to-video alignment
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
