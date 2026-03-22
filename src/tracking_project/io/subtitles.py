"""
subtitles.py

This module parses subtitle files into structured time-stamped segments for
coarse script-to-video alignment.

Architectural role
------------------
The coarse alignment layer needs a time-ordered representation of the movie's
dialogue evidence.
This module provides that representation in a deterministic, reusable format.

The current implementation targets subtitle formats that expose explicit time
intervals line by line:
    - `.srt`
    - `.vtt`
    - normalized subtitle `.json`

No narrative reasoning happens here. This module only converts subtitle text
into normalized `SubtitleSegment` objects.
"""
from __future__ import annotations

import re
import json
from dataclasses import dataclass
from typing import List


_TIMECODE_RE = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2}[,\.]\d{3})"
)


@dataclass(frozen=True)
class SubtitleSegment:
    """
    One subtitle interval with normalized timing and text.
    """

    index: int
    start_sec: float
    end_sec: float
    text: str

    @property
    def center_sec(self) -> float:
        return 0.5 * (self.start_sec + self.end_sec)


def _parse_timecode(raw: str) -> float:
    hh, mm, ss_ms = raw.replace(".", ",").split(":")
    ss, ms = ss_ms.split(",")
    return (
        int(hh) * 3600
        + int(mm) * 60
        + int(ss)
        + int(ms) / 1000.0
    )


def _load_subtitle_segments_from_json(path: str) -> List[SubtitleSegment]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    segments = obj.get("segments", [])
    out: List[SubtitleSegment] = []
    for idx, rec in enumerate(segments):
        text = str(rec.get("text", "")).strip()
        if not text:
            continue
        out.append(
            SubtitleSegment(
                index=int(rec.get("index", idx)),
                start_sec=float(rec["start_sec"]),
                end_sec=float(rec["end_sec"]),
                text=text,
            )
        )
    return out


def load_subtitle_segments(path: str) -> List[SubtitleSegment]:
    """
    Load subtitle segments from an `.srt`, `.vtt`, or normalized `.json` file.

    Args:
        path: Subtitle file path.

    Returns:
        Ordered list of `SubtitleSegment`.
    """
    if path.lower().endswith(".json"):
        return _load_subtitle_segments_from_json(path)

    with open(path, "r", encoding="utf-8-sig") as f:
        raw_text = f.read()

    blocks = re.split(r"\r?\n\r?\n+", raw_text.strip())
    segments: List[SubtitleSegment] = []

    for block in blocks:
        lines = [line.strip("\ufeff") for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        time_idx = None
        for idx, line in enumerate(lines):
            if _TIMECODE_RE.search(line):
                time_idx = idx
                break

        if time_idx is None:
            continue

        match = _TIMECODE_RE.search(lines[time_idx])
        if match is None:
            continue

        start_sec = _parse_timecode(match.group("start"))
        end_sec = _parse_timecode(match.group("end"))
        text = " ".join(lines[time_idx + 1 :]).strip()
        if not text:
            continue

        segments.append(
            SubtitleSegment(
                index=len(segments),
                start_sec=start_sec,
                end_sec=end_sec,
                text=text,
            )
        )

    return segments
