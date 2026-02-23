#!/usr/bin/env python
import argparse
import os

from tracking_project.pipeline.stage1_mine import mine_stage1_llm


def main() -> None:
    """
    Command-line entry point for Stage-1 LLM-first mining.

    This script is intentionally thin: all the real work happens in the
    tracking_project.pipeline.stage1_mine.mine_stage1_llm() function.
    """
    ap = argparse.ArgumentParser(description="Stage-1 LLM-first narrative re-occurrence mining.")
    ap.add_argument("--pdf", required=True, help="Path to the screenplay PDF.")
    ap.add_argument("--out", required=True, help="Path to the output events JSON file.")

    ap.add_argument(
        "--llm_model",
        default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        help="OpenAI model name to use for scene-level extraction.",
    )
    ap.add_argument(
        "--cache_dir",
        default="data/llm_cache",
        help="Root directory for LLM cache files.",
    )
    ap.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Maximum number of events to keep in the final JSON.",
    )
    ap.add_argument(
        "--min_gap",
        type=int,
        default=20,
        help="Minimum required gap_jump for an entity to be considered.",
    )
    ap.add_argument(
        "--keep_categories",
        default="object,artifact,weapon,creature,vehicle",
        help="Comma-separated list of allowed LLM categories.",
    )
    ap.add_argument("--min_conf", type=float, default=0.70)
    ap.add_argument("--min_visual", type=float, default=0.60)
    ap.add_argument("--min_action", type=float, default=0.45)
    ap.add_argument("--min_state", type=float, default=0.25)

    ap.add_argument(
        "--debug_scene_parse",
        action="store_true",
        help="If set, dump additional scene parsing debug info.",
    )
    ap.add_argument(
        "--dump_scenes_path",
        default="data/processed/scene_markers.json",
        help="Where to dump a compact scene marker summary.",
    )

    args = ap.parse_args()
    keep_categories = [x.strip() for x in args.keep_categories.split(",") if x.strip()]

    mine_stage1_llm(
        pdf=args.pdf,
        out=args.out,
        llm_model=args.llm_model,
        cache_dir=args.cache_dir,
        top_k=args.top_k,
        min_gap=args.min_gap,
        keep_categories=keep_categories,
        min_conf=args.min_conf,
        min_visual=args.min_visual,
        min_action=args.min_action,
        min_state=args.min_state,
        debug_scene_parse=args.debug_scene_parse,
        dump_scenes_path=args.dump_scenes_path,
    )


if __name__ == "__main__":
    main()