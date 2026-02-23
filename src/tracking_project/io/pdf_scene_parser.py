"""
pdf_scene_parser.py

This module parses a screenplay PDF into structured Scene objects.

It is the structural backbone of Stage-1 narrative re-occurrence mining.

High-level purpose
------------------
Given a screenplay PDF, this module:

1) Detects scene heading markers (e.g., "12 INT. GREAT HALL - NIGHT")
2) Splits the screenplay into Scene objects
3) Assigns two types of scene indices:
   - scene_idx: counts all scene markers including OMITTED
   - scene_idx_non_omitted: counts only real scenes (OMITTED → -1)
4) Collects the raw lines belonging to each scene
5) Records page ranges for each scene

The output of this module is purely structural and deterministic.
It does NOT perform any semantic processing.

This module remain simple, deterministic, and well-tested. (Passed tests in 
test_scene_parser.py with a variety of edge cases.)
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import pdfplumber


def clean_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


@dataclass(frozen=True)
class Scene:
    """
    scene_idx: sequential index in screenplay marker order (0..N-1), includes OMITTED markers
    scene_idx_non_omitted: sequential index counting ONLY non-OMITTED markers (0..~115), OMITTED -> -1
    scene_label_raw: printed label (e.g., "12", "12A")
    scene_label_int: numeric part of label (e.g., 12)
    marker_type: normalized (INT., EXT., INT/EXT., EXT/INT., OMITTED)
    """
    scene_idx: int
    scene_idx_non_omitted: int
    scene_label_raw: str
    scene_label_int: int
    marker_type: str
    heading: str
    page_start: int
    page_end: int
    lines: List[str]


# More permissive scene heading regex:
# - supports 12A, 12B
# - supports "EXT. /INT." with spaces around slash
# - supports INT/EXT without periods
SCENE_MARK_RE = re.compile(
    r"""^\s*
    (?P<label>\d{1,3}[A-Z]?)\.?
    \s+
    (?P<type>
        OMITTED |
        (?:INT|EXT)\.?\s*/\s*(?:INT|EXT)\.? |
        INT\.? | EXT\.?
    )
    \s*
    (?P<rest>.*)$
    """,
    re.VERBOSE,
)


def normalize_marker_type(raw_type: str) -> str:
    t = clean_spaces(raw_type)
    t = re.sub(r"\s*/\s*", "/", t)  # normalize slash spacing
    up = t.upper()

    if up == "OMITTED":
        return "OMITTED"

    # Remove periods to normalize variants
    up = up.replace(".", "")

    if up == "INT":
        return "INT."
    if up == "EXT":
        return "EXT."
    if up == "INT/EXT":
        return "INT/EXT."
    if up == "EXT/INT":
        return "EXT/INT."

    # Fallback (should rarely happen)
    return t


def _strip_trailing_scene_label(rest: str, scene_label_int: int) -> str:
    r = clean_spaces(rest)
    # headings sometimes end with " ... 12"
    r = re.sub(rf"\s+{scene_label_int}\s*$", "", r)
    return r.strip()


def parse_pdf_scenes(
    pdf_path: str,
    *,
    debug: bool = False,
    dump_path: Optional[str] = None,
    pdf_open: Callable[..., Any] = pdfplumber.open,
) -> Tuple[List[Scene], Dict]:
    """
    Parse screenplay PDF into scenes.

    Returns (scenes, meta).
    meta includes near_miss examples to debug undercounting.
    """

    pages: List[str] = []
    with pdf_open(pdf_path) as pdf:
        for p in pdf.pages:
            pages.append(p.extract_text() or "")

    raw_scenes = []
    cur = None  # [label_raw, label_int, marker_type, heading, ps, pe, lines]

    near_miss = []
    heading_like = re.compile(r"^\s*\d{1,3}[A-Z]?\s+.*\b(INT|EXT)\b", re.I)

    def flush():
        nonlocal cur
        if cur is not None:
            raw_scenes.append(cur)
            cur = None

    for pno, txt in enumerate(pages, start=1):
        for raw in txt.splitlines():
            s = raw.strip()
            if not s:
                if cur is not None:
                    cur[6].append("")
                continue

            m = SCENE_MARK_RE.match(s)
            if m:
                flush()
                label_raw = m.group("label")
                label_int = int(re.match(r"\d{1,3}", label_raw).group(0))
                marker_type = normalize_marker_type(m.group("type"))
                rest = _strip_trailing_scene_label(m.group("rest"), label_int)
                heading = clean_spaces(f"{marker_type} {rest}".strip())
                cur = [label_raw, label_int, marker_type, heading, pno, pno, []]
                continue

            if debug and heading_like.match(s) and not SCENE_MARK_RE.match(s):
                if len(near_miss) < 60:
                    near_miss.append({"page": pno, "line": s})

            if cur is not None:
                cur[5] = pno
                if s.startswith("Rev.") or s.startswith("HARRY POTTER"):
                    continue
                cur[6].append(raw.rstrip("\n"))

    flush()

    scenes: List[Scene] = []
    non_omitted_idx = 0
    for idx, (label_raw, label_int, marker_type, heading, ps, pe, lines) in enumerate(raw_scenes):
        cur_non_omitted = non_omitted_idx if marker_type != "OMITTED" else -1
        if marker_type != "OMITTED":
            non_omitted_idx += 1

        scenes.append(
            Scene(
                scene_idx=idx,
                scene_idx_non_omitted=cur_non_omitted,
                scene_label_raw=label_raw,
                scene_label_int=label_int,
                marker_type=marker_type,
                heading=heading,
                page_start=ps,
                page_end=pe,
                lines=lines,
            )
        )

    meta = {
        "scene_markers_total": len(scenes),
        "scene_markers_non_omitted": sum(1 for s in scenes if s.marker_type != "OMITTED"),
        "near_miss_count": len(near_miss),
        "near_miss_examples": near_miss,
    }

    if dump_path:
        os.makedirs(os.path.dirname(dump_path), exist_ok=True)
        dump = {
            "meta": meta,
            "scenes": [
                {
                    "scene_idx": s.scene_idx,
                    "scene_idx_non_omitted": s.scene_idx_non_omitted,
                    "scene_label_raw": s.scene_label_raw,
                    "scene_label_int": s.scene_label_int,
                    "marker_type": s.marker_type,
                    "heading": s.heading,
                    "page_start": s.page_start,
                    "page_end": s.page_end,
                }
                for s in scenes
            ],
        }
        with open(dump_path, "w", encoding="utf-8") as f:
            json.dump(dump, f, indent=2)

    return scenes, meta