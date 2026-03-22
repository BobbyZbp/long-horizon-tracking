"""
test_subtitles.py

Unit tests for deterministic subtitle parsing used by the coarse alignment
layer.
"""
from __future__ import annotations

from tracking_project.io.subtitles import load_subtitle_segments


def test_load_subtitle_segments_parses_basic_srt(tmp_path):
    subtitle_path = tmp_path / "sample.srt"
    subtitle_path.write_text(
        "\n".join(
            [
                "1",
                "00:00:01,500 --> 00:00:03,000",
                "Hello there.",
                "",
                "2",
                "00:00:05,000 --> 00:00:07,250",
                "Bring the diary.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    segments = load_subtitle_segments(str(subtitle_path))

    assert len(segments) == 2
    assert segments[0].start_sec == 1.5
    assert segments[0].end_sec == 3.0
    assert segments[0].text == "Hello there."
    assert segments[1].text == "Bring the diary."


def test_load_subtitle_segments_parses_normalized_json(tmp_path):
    subtitle_path = tmp_path / "sample.json"
    subtitle_path.write_text(
        """
{
  "segments": [
    {"index": 7, "start_sec": 12.5, "end_sec": 15.2, "text": "Did you bring the diary?"},
    {"index": 8, "start_sec": 16.0, "end_sec": 18.0, "text": "No."}
  ]
}
        """.strip(),
        encoding="utf-8",
    )

    segments = load_subtitle_segments(str(subtitle_path))

    assert len(segments) == 2
    assert segments[0].index == 7
    assert segments[0].start_sec == 12.5
    assert segments[1].text == "No."
