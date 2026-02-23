from __future__ import annotations

import re
from typing import List

# Patterns for screenplay layout noise that should not be treated as narrative text.
_JUNK_LINE_RE = re.compile(
    r"""(
        ^\s*\(CONTINUED\)       |  # (CONTINUED)
        \bCONTINUED:\b          |  # CONTINUED:
        ^\s*FADE\s+(IN|OUT)\b   |  # FADE IN / FADE OUT
        ^\s*CUT\s+TO\b          |  # CUT TO
        ^\s*DISSOLVE\s+TO\b     |  # DISSOLVE TO
        ^\s*Rev\.\s*            |  # revision header lines starting with 'Rev.'
        ^HARRY\s+POTTER\b       |  # typical header for the HP screenplay
        ^\s*PAGE\s+\d+\s*$         # page footer like 'PAGE 12'
    )""",
    re.IGNORECASE | re.VERBOSE,
)


def clean_scene_lines(lines: List[str]) -> List[str]:
    """
    Remove script layout boilerplate from a scene's raw lines while preserving
    the basic block structure (blank lines are kept).

    This function is a light pre-processing step before block splitting:
    - It drops continuation markers, revision headers, page headers/footers,
      and other direction-like boilerplate that should not feed the LLM.
    - It keeps all non-junk lines and all blank lines, to allow the existing
      split_blocks() logic to correctly separate action vs dialogue blocks.

    Args:
        lines: Raw lines from a Scene, as returned by pdf_scene_parser.

    Returns:
        A new list of lines where obvious layout junk has been removed.
        The number and positions of blank lines may change but the relative
        grouping of narrative content is preserved.
    """
    cleaned: List[str] = []
    for ln in lines:
        if not ln.strip():
            # Preserve blank lines as block separators
            cleaned.append(ln)
            continue

        if _JUNK_LINE_RE.search(ln):
            # Drop layout/junk lines
            continue

        cleaned.append(ln)
    return cleaned