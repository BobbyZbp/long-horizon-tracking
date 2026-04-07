#!/usr/bin/env python
"""
mine_climax_llm.py

Score each non-omitted scene for narrative climax intensity.

What this file does
-------------------
- parses the screenplay into scenes
- derives a scene-level object map from per-scene extraction rather than only
  from filtered Stage-1 events
- queries the LLM for a climax score per scene

Inputs
------
- `--pdf`: screenplay PDF
- `--events`: existing Stage-1 events JSON used for compatibility/context
- `--out`: destination JSON path
- optional LLM model and cache directory settings

Outputs
-------
- writes one scene-level climax JSON file, typically `scene_climax.json`
- output records include each scene heading, climax score, and extracted
  scene-level objects

Workflow position
-----------------
- upstream: screenplay parsing and per-scene extraction
- current stage: climax scoring
- downstream: Chekhov candidate validation, where later high-climax scenes are
  used as payoff evidence
"""

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tracking_project.pipeline.climax_mine import mine_climax_llm


def main():
    ap = argparse.ArgumentParser(
        description="Scene-level climax scoring."
    )
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--llm_model",
        default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    )
    ap.add_argument(
        "--cache_dir",
        default="data/llm_cache",
    )

    args = ap.parse_args()

    mine_climax_llm(
        pdf=args.pdf,
        events=args.events,
        out=args.out,
        llm_model=args.llm_model,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
