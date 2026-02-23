"""
blocks.py

This module splits a scene's raw lines into structural blocks and classifies
each block as either "action" or "dialogue".

Purpose in the pipeline
-----------------------
After pdf_scene_parser constructs Scene objects, each scene contains raw lines.
However, screenplay text interleaves:

    - ACTION description (narrative text)
    - CHARACTER names
    - DIALOGUE lines
    - Direction markers (CUT TO, FADE IN, etc.)

For LLM-first mining, we only feed ACTION blocks to the LLM.
Dialogue is excluded to:
    - reduce noise
    - avoid psychological/abstract mentions
    - focus on visually grounded entities

This module performs:

1) split_blocks:
    Groups consecutive non-empty lines into blocks.
2) block_type:
    Classifies each block as "dialogue" or "action"
    based on the first line of the block.

This module is deterministic and contains no LLM logic.
It defines what text is considered "visual narrative".
"""
import re
from typing import List

SPEAKER_RE = re.compile(r"^[A-Z][A-Z \-'\.]+$")
DIRECTION_CAPS = {"CUT TO", "DISSOLVE TO", "FADE IN", "FADE OUT", "MONTAGE", "OMITTED"}


def _clean_spaces(s: str) -> str:
    return " ".join(s.split()).strip()


def split_blocks(lines: List[str]) -> List[List[str]]:
    blocks: List[List[str]] = []
    cur: List[str] = []
    for ln in lines:
        if ln.strip() == "":
            if cur:
                blocks.append(cur)
                cur = []
        else:
            cur.append(ln)
    if cur:
        blocks.append(cur)
    return blocks


def is_speaker_line(s: str) -> bool:
    s = _clean_spaces(s)
    if len(s) < 2 or len(s) > 32:
        return False
    if any(c.islower() for c in s):
        return False
    if s in DIRECTION_CAPS:
        return False
    return bool(SPEAKER_RE.match(s))


def block_type(block: List[str]) -> str:
    head = _clean_spaces(block[0]) if block else ""
    return "dialogue" if is_speaker_line(head) else "action"