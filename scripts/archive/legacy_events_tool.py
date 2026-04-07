#!/usr/bin/env python3
"""
legacy_events_tool.py

Archived utility for the older heuristic event-mining workflow.

What this file does
-------------------
- `mine`: runs the older lightweight event miner based on simple scene parsing
  and speaker-like heuristics
- `inspect`: loads an events JSON file and prints, filters, or exports a CSV
  summary for manual review

Inputs
------
- `mine` expects a screenplay PDF plus an output path and optional top-k limit
- `inspect` expects an events JSON file plus optional entity/category/gap
  filters and CSV export settings

Outputs
-------
- `mine` writes a lightweight events JSON in the older exploratory schema
- `inspect` prints event summaries to stdout and can optionally export a CSV

Workflow position
-----------------
- this is not part of the canonical narrative-to-video pipeline
- it is preserved only for historical comparison, quick heuristic baselines,
  and manual inspection of older event outputs
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

import pdfplumber


HEADING_RE = re.compile(r"^\s*(\d+)\s+(INT\.|EXT\.|INT/EXT\.|EXT/INT\.)\s+(.+)$")
SPEAKER_RE = re.compile(r"^(?P<name>[A-Z][A-Z \-'\.]+?)(?:\s*\(.*\))?\s*$")


def clean(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip(" ,.;:()[]")


def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_scenes(pdf_path: str) -> List[Dict[str, Any]]:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")

    scenes: List[Dict[str, Any]] = []
    cur: Dict[str, Any] | None = None

    def flush() -> None:
        nonlocal cur
        if cur:
            cur["text"] = "\n".join(cur["lines"]).strip()
            del cur["lines"]
            scenes.append(cur)
            cur = None

    for pno, txt in enumerate(pages, start=1):
        for line in txt.splitlines():
            line_stripped = line.strip()

            match = HEADING_RE.match(line_stripped)
            if match:
                flush()
                cur = {
                    "scene_no": int(match.group(1)),
                    "heading": clean(f"{match.group(2)} {match.group(3)}"),
                    "page_start": pno,
                    "page_end": pno,
                    "lines": [line_stripped],
                }
                continue

            if cur is not None:
                cur["page_end"] = pno
                if line_stripped.startswith("THE CHAMBER OF SECRETS"):
                    continue
                cur["lines"].append(line.rstrip())

    flush()
    scenes.sort(key=lambda x: x["scene_no"])
    return scenes


def extract_entities(scene_text: str) -> set[str]:
    entities: set[str] = set()
    for line in scene_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if HEADING_RE.match(line):
            continue
        match = SPEAKER_RE.match(line)
        if match:
            name = clean(match.group("name").replace(".", ""))
            if 1 <= len(name) <= 22 and not name.startswith(("INT", "EXT")):
                entities.add(name)
    return entities


def max_gap_pair(seq: List[int]) -> tuple[int, int, int] | None:
    seq = sorted(set(seq))
    best = None
    for a, b in zip(seq, seq[1:]):
        gap = b - a
        if best is None or gap > best[0]:
            best = (gap, a, b)
    return best


def norm_ent(e: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", e.lower())


def mine_events(scenes: List[Dict[str, Any]], k: int = 30) -> List[Dict[str, Any]]:
    scene_by_no = {s["scene_no"]: s for s in scenes}
    ent2scenes: Dict[str, List[int]] = defaultdict(list)

    for scene in scenes:
        ents = extract_entities(scene["text"])
        for ent in ents:
            ent2scenes[ent].append(scene["scene_no"])

    events: List[Dict[str, Any]] = []
    for ent, seq in ent2scenes.items():
        if len(set(seq)) < 2:
            continue
        pair = max_gap_pair(seq)
        if pair is None:
            continue
        gap, a, b = pair
        events.append(
            {
                "entity": ent,
                "gap_scenes": gap,
                "scene_a": a,
                "scene_b": b,
                "page_a": scene_by_no[a]["page_start"],
                "page_b": scene_by_no[b]["page_start"],
                "heading_a": scene_by_no[a]["heading"],
                "heading_b": scene_by_no[b]["heading"],
            }
        )

    events.sort(key=lambda x: (-x["gap_scenes"], x["entity"]))

    out: List[Dict[str, Any]] = []
    seen = set()
    for ev in events:
        key = norm_ent(ev["entity"])
        if key in seen:
            continue
        seen.add(key)
        out.append(ev)
        if len(out) >= k:
            break
    return out


def _safe_get(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return default


def event_gap(event: Dict[str, Any]) -> int:
    g = _safe_get(event, ["gap_jump", "gap_scenes_non_omitted", "gap_scenes"], None)
    try:
        return int(g)
    except Exception:
        return -1


def event_category(event: Dict[str, Any]) -> str:
    return str(_safe_get(event, ["entity_type", "entity_category", "type"], "unknown"))


def event_name(event: Dict[str, Any]) -> str:
    return str(_safe_get(event, ["canonical_name", "entity", "entity_key"], "unknown"))


def summarize_event(event: Dict[str, Any]) -> Dict[str, Any]:
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

    centrality = _safe_get(scores_obj, ["narrative_action_centrality"], llm_obj.get("narrative_action_centrality"))
    visual = _safe_get(scores_obj, ["visual_grounded_score"], llm_obj.get("visual_grounded_score"))
    state = _safe_get(scores_obj, ["state_change_potential"], llm_obj.get("state_change_potential"))
    conf = _safe_get(scores_obj, ["confidence"], llm_obj.get("confidence"))

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


def _run_mine(args: argparse.Namespace) -> None:
    scenes = extract_scenes(args.pdf)
    events = mine_events(scenes, k=args.k)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"pdf": args.pdf, "k": args.k, "events": events}, f, indent=2)

    print(f"[ok] scenes={len(scenes)} events={len(events)} -> {args.out}")


def _run_inspect(args: argparse.Namespace) -> None:
    obj = load_json(args.file)
    events = obj.get("events", [])
    if not isinstance(events, list):
        raise RuntimeError("JSON file does not contain a list under key 'events'.")

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
        counter = Counter([normalize(event_category(e)) for e in events])
        for k, v in counter.most_common():
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
    filtered.sort(key=lambda e: (event_gap(e), str(event_name(e))), reverse=True)

    if args.csv:
        export_csv(filtered, args.csv)
        print(f"[ok] wrote CSV: {args.csv}")

    show_context = not args.no_context
    for ev in filtered[: args.top]:
        print_event(ev, show_context=show_context)

    if not filtered:
        print("[info] No events matched your filters.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Archived utility for the older heuristic event mining and inspection workflow.",
    )
    sub = ap.add_subparsers(dest="mode", required=True)

    mine_parser = sub.add_parser("mine", help="Run the older heuristic event miner.")
    mine_parser.add_argument("--pdf", required=True)
    mine_parser.add_argument("--k", type=int, default=30)
    mine_parser.add_argument("--out", required=True)
    mine_parser.set_defaults(handler=_run_mine)

    inspect_parser = sub.add_parser("inspect", help="Inspect historical Stage-1 style events JSON.")
    inspect_parser.add_argument("--file", required=True, help="Path to events JSON file.")
    inspect_parser.add_argument("--top", type=int, default=20, help="Number of events to print after filtering.")
    inspect_parser.add_argument("--entity", default=None, help="Filter: canonical_name contains this substring (case-insensitive).")
    inspect_parser.add_argument("--category", default=None, help="Filter: exact category/type match (case-insensitive).")
    inspect_parser.add_argument("--min_gap", type=int, default=None, help="Filter: minimum gap.")
    inspect_parser.add_argument("--max_gap", type=int, default=None, help="Filter: maximum gap.")
    inspect_parser.add_argument("--keyword", default=None, help="Filter: keyword must appear in context or name.")
    inspect_parser.add_argument("--no_context", action="store_true", help="Do not print contexts.")
    inspect_parser.add_argument("--csv", default=None, help="If set, export filtered summary to CSV path.")
    inspect_parser.add_argument("--list_categories", action="store_true", help="Print category counts and exit.")
    inspect_parser.add_argument("--stats", action="store_true", help="Print quick dataset stats and exit.")
    inspect_parser.set_defaults(handler=_run_inspect)

    args = ap.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
