#!/usr/bin/env python3
"""
Inspect Stage-1 mined events JSON.

This script is intentionally simple and "human-in-the-loop":
- Print the top-N events with key fields and context.
- Filter by keyword/category/gap range.
- Drill into a specific entity (canonical_name contains substring).
- Export a compact CSV summary (optional).

Use cases:
1) Quick quality check after running mining:
   python scripts/inspect_events.py --file data/events/events_top100.json --top 20

2) Find all events for something like "diary":
   python scripts/inspect_events.py --file data/events/events_top100.json --entity diary --top 50

3) Filter by category and minimum gap:
   python scripts/inspect_events.py --file data/events/events_top100.json --category artifact --min_gap 30 --top 30

4) Show only fields (no context) for fast scanning:
   python scripts/inspect_events.py --file data/events/events_top100.json --top 50 --no_context
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple


def load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file into a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_get(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    """Try multiple keys in order; return the first that exists."""
    for k in keys:
        if k in d:
            return d[k]
    return default


def normalize(s: str) -> str:
    """Lowercase + collapse whitespace for loose matching."""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def event_gap(event: Dict[str, Any]) -> int:
    """
    Return the gap value used for ranking.

    Preferred:
      - gap_jump (Stage-1 LLM-first output)
    Fallbacks:
      - gap_scenes, gap_scenes_non_omitted
    """
    g = _safe_get(event, ["gap_jump", "gap_scenes_non_omitted", "gap_scenes"], None)
    try:
        return int(g)
    except Exception:
        return -1


def event_category(event: Dict[str, Any]) -> str:
    """Return event category/type."""
    return str(_safe_get(event, ["entity_type", "entity_category", "type"], "unknown"))


def event_name(event: Dict[str, Any]) -> str:
    """Return canonical name if available, else entity/entity_key."""
    return str(_safe_get(event, ["canonical_name", "entity", "entity_key"], "unknown"))


def summarize_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a compact summary dict for printing/CSV."""
    name = event_name(event)
    cat = event_category(event)
    gap = event_gap(event)

    scene_a = _safe_get(event, ["scene_a_idx_non_omitted", "scene_a", "scene_a_idx"], None)
    scene_b = _safe_get(event, ["scene_b_idx_non_omitted", "scene_b", "scene_b_idx"], None)

    page_a = _safe_get(event, ["page_a"], None)
    page_b = _safe_get(event, ["page_b"], None)

    heading_a = _safe_get(event, ["heading_a"], "")
    heading_b = _safe_get(event, ["heading_b"], "")

    score = _safe_get(event, ["score"], None)

    scores_obj = event.get("scores", {}) if isinstance(event.get("scores", {}), dict) else {}
    llm_obj = event.get("llm_curator", {}) if isinstance(event.get("llm_curator", {}), dict) else {}

    # allow both schema variants
    centrality = _safe_get(scores_obj, ["narrative_action_centrality"], None)
    visual = _safe_get(scores_obj, ["visual_grounded_score"], None)
    state = _safe_get(scores_obj, ["state_change_potential"], None)
    conf = _safe_get(scores_obj, ["confidence"], None)

    if centrality is None:
        centrality = llm_obj.get("narrative_action_centrality")
    if visual is None:
        visual = llm_obj.get("visual_grounded_score")
    if state is None:
        state = llm_obj.get("state_change_potential")
    if conf is None:
        conf = llm_obj.get("confidence")

    return {
        "name": name,
        "category": cat,
        "gap": gap,
        "scene_a": scene_a,
        "scene_b": scene_b,
        "page_a": page_a,
        "page_b": page_b,
        "heading_a": heading_a,
        "heading_b": heading_b,
        "score": score,
        "action": centrality,
        "visual": visual,
        "state": state,
        "conf": conf,
    }


def matches_filters(
    event: Dict[str, Any],
    *,
    entity_substr: Optional[str],
    category: Optional[str],
    min_gap: Optional[int],
    max_gap: Optional[int],
    keyword: Optional[str],
) -> bool:
    """Return True if event passes all requested filters."""
    name = normalize(event_name(event))
    cat = normalize(event_category(event))
    gap = event_gap(event)

    if entity_substr and normalize(entity_substr) not in name:
        return False
    if category and normalize(category) != cat:
        return False
    if min_gap is not None and gap < min_gap:
        return False
    if max_gap is not None and gap > max_gap:
        return False

    if keyword:
        kw = normalize(keyword)
        ctx_a = _safe_get(event, ["context_a"], [])
        ctx_b = _safe_get(event, ["context_b"], [])
        all_ctx = " ".join([str(x) for x in (ctx_a or []) + (ctx_b or [])])
        if kw not in normalize(all_ctx) and kw not in name:
            return False

    return True


def print_event(event: Dict[str, Any], *, show_context: bool = True) -> None:
    """Pretty-print one event to stdout."""
    s = summarize_event(event)
    print("=" * 80)
    print(f"{s['name']}  |  category={s['category']}  |  gap={s['gap']}  |  score={s['score']}")
    print(f"Scene A: idx={s['scene_a']}  page={s['page_a']}  heading={s['heading_a']}")
    print(f"Scene B: idx={s['scene_b']}  page={s['page_b']}  heading={s['heading_b']}")
    print(f"Scores: action={s['action']}  visual={s['visual']}  state={s['state']}  conf={s['conf']}")

    if not show_context:
        return

    ctx_a = _safe_get(event, ["context_a"], []) or []
    ctx_b = _safe_get(event, ["context_b"], []) or []
    if ctx_a:
        print("\nContext A:")
        for line in ctx_a[:6]:
            print(f"  - {line}")
    if ctx_b:
        print("\nContext B:")
        for line in ctx_b[:6]:
            print(f"  - {line}")


def export_csv(events: List[Dict[str, Any]], out_csv: str) -> None:
    """Export a compact CSV summary for quick scanning."""
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    fieldnames = [
        "name", "category", "gap", "scene_a", "scene_b",
        "page_a", "page_b", "heading_a", "heading_b",
        "score", "action", "visual", "state", "conf"
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for ev in events:
            w.writerow(summarize_event(ev))


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect mined Stage-1 events JSON.")
    ap.add_argument("--file", required=True, help="Path to events JSON file.")
    ap.add_argument("--top", type=int, default=20, help="Number of events to print after filtering.")
    ap.add_argument("--entity", default=None, help="Filter: canonical_name contains this substring (case-insensitive).")
    ap.add_argument("--category", default=None, help="Filter: exact category/type match (case-insensitive).")
    ap.add_argument("--min_gap", type=int, default=None, help="Filter: minimum gap.")
    ap.add_argument("--max_gap", type=int, default=None, help="Filter: maximum gap.")
    ap.add_argument("--keyword", default=None, help="Filter: keyword must appear in context or name.")
    ap.add_argument("--no_context", action="store_true", help="Do not print contexts.")
    ap.add_argument("--csv", default=None, help="If set, export filtered summary to CSV path.")
    ap.add_argument("--list_categories", action="store_true", help="Print category counts and exit.")
    ap.add_argument("--stats", action="store_true", help="Print quick dataset stats and exit.")

    args = ap.parse_args()
    obj = load_json(args.file)

    events = obj.get("events", [])
    if not isinstance(events, list):
        raise RuntimeError("JSON file does not contain a list under key 'events'.")

    # quick stats
    if args.stats:
        gaps = [event_gap(e) for e in events if event_gap(e) >= 0]
        cats = [event_category(e) for e in events]
        print(f"Events: {len(events)}")
        if gaps:
            print(f"Gap: min={min(gaps)}  median={sorted(gaps)[len(gaps)//2]}  max={max(gaps)}")
        print(f"Categories: {len(set(cats))}")
        return

    if args.list_categories:
        from collections import Counter
        c = Counter([normalize(event_category(e)) for e in events])
        for k, v in c.most_common():
            print(f"{k}: {v}")
        return

    filtered = [
        e for e in events
        if matches_filters(
            e,
            entity_substr=args.entity,
            category=args.category,
            min_gap=args.min_gap,
            max_gap=args.max_gap,
            keyword=args.keyword,
        )
    ]

    # Stable sort by gap desc as a default for inspection
    filtered.sort(key=lambda e: (event_gap(e), str(event_name(e))), reverse=True)

    if args.csv:
        export_csv(filtered, args.csv)
        print(f"[ok] wrote CSV: {args.csv}")

    show_context = not args.no_context
    for ev in filtered[: args.top]:
        print_event(ev, show_context=show_context)

    if not filtered:
        print("[info] No events matched your filters.")


if __name__ == "__main__":
    main()