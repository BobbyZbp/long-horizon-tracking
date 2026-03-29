#!/usr/bin/env python

import argparse
import os

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