# scripts/mine_events.py
import re, json, argparse
from collections import defaultdict

import pdfplumber

HEADING_RE = re.compile(r'^\s*(\d+)\s+(INT\.|EXT\.|INT/EXT\.|EXT/INT\.)\s+(.+)$')
SPEAKER_RE = re.compile(r'^(?P<name>[A-Z][A-Z \-\'\.]+?)(?:\s*\(.*\))?\s*$')

def clean(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip(" ,.;:()[]")

def extract_scenes(pdf_path: str):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            pages.append(p.extract_text() or "")

    scenes = []
    cur = None

    def flush():
        nonlocal cur
        if cur:
            cur["text"] = "\n".join(cur["lines"]).strip()
            del cur["lines"]
            scenes.append(cur)
            cur = None

    for pno, txt in enumerate(pages, start=1):
        for line in txt.splitlines():
            line_stripped = line.strip()

            m = HEADING_RE.match(line_stripped)
            if m:
                flush()
                cur = {
                    "scene_no": int(m.group(1)),
                    "heading": clean(f"{m.group(2)} {m.group(3)}"),
                    "page_start": pno,
                    "page_end": pno,
                    "lines": [line_stripped],
                }
                continue

            if cur is not None:
                cur["page_end"] = pno
                # drop headers like "THE CHAMBER OF SECRETS - Rev..."
                if line_stripped.startswith("THE CHAMBER OF SECRETS"):
                    continue
                cur["lines"].append(line.rstrip())

    flush()
    scenes.sort(key=lambda x: x["scene_no"])
    return scenes

def extract_entities(scene_text: str):
    entities = set()
    for line in scene_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if HEADING_RE.match(line):
            continue
        m = SPEAKER_RE.match(line)
        if m:
            name = clean(m.group("name").replace(".", ""))
            if 1 <= len(name) <= 22 and not name.startswith(("INT", "EXT")):
                entities.add(name)
    return entities

def max_gap_pair(seq):
    seq = sorted(set(seq))
    best = None
    for a, b in zip(seq, seq[1:]):
        gap = b - a
        if best is None or gap > best[0]:
            best = (gap, a, b)
    return best  # (gap, a, b)

def norm_ent(e: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", e.lower())

def mine_events(scenes, k=30):
    scene_by_no = {s["scene_no"]: s for s in scenes}
    ent2scenes = defaultdict(list)

    for s in scenes:
        ents = extract_entities(s["text"])
        for e in ents:
            ent2scenes[e].append(s["scene_no"])

    events = []
    for e, seq in ent2scenes.items():
        if len(set(seq)) < 2:
            continue
        gap, a, b = max_gap_pair(seq)
        events.append({
            "entity": e,
            "gap_scenes": gap,
            "scene_a": a,
            "scene_b": b,
            "page_a": scene_by_no[a]["page_start"],
            "page_b": scene_by_no[b]["page_start"],
            "heading_a": scene_by_no[a]["heading"],
            "heading_b": scene_by_no[b]["heading"],
        })

    events.sort(key=lambda x: (-x["gap_scenes"], x["entity"]))

    # keep unique entities by normalized key
    out, seen = [], set()
    for ev in events:
        key = norm_ent(ev["entity"])
        if key in seen:
            continue
        seen.add(key)
        out.append(ev)
        if len(out) >= k:
            break
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    scenes = extract_scenes(args.pdf)
    events = mine_events(scenes, k=args.k)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"pdf": args.pdf, "k": args.k, "events": events}, f, indent=2)

    print(f"[ok] scenes={len(scenes)} events={len(events)} -> {args.out}")

if __name__ == "__main__":
    main()