"""
test_video_alignment.py

Unit tests for coarse script-to-video alignment.
"""
from __future__ import annotations

from tracking_project.io.subtitles import SubtitleSegment
from tracking_project.pipeline.video_alignment import build_script_video_alignment


def test_build_script_video_alignment_uses_order_and_subtitle_overlap():
    candidates_json = {
        "candidates": [
            {
                "object_id": "tom riddles diary",
                "canonical_name": "Tom Riddle's diary",
                "candidate_source": "inventory_with_stage1_prior",
                "loose_scene_chain": [
                    {
                        "scene_idx_non_omitted": 2,
                        "heading": "INT. DORMITORY - NIGHT",
                        "matched_inventory_objects": ["diary"],
                        "inventory_evidence": ["Harry opens the diary."],
                    },
                    {
                        "scene_idx_non_omitted": 10,
                        "heading": "INT. CHAMBER - NIGHT",
                        "matched_inventory_objects": ["Tom Riddle's diary"],
                        "inventory_evidence": ["Destroy the diary."],
                    },
                ],
            }
        ]
    }
    scene_inventory_json = {
        "meta": {"scenes_non_omitted": 12},
        "scenes": {},
    }
    subtitle_segments = [
        SubtitleSegment(index=0, start_sec=0.0, end_sec=5.0, text="Welcome back to Hogwarts."),
        SubtitleSegment(index=1, start_sec=35.0, end_sec=40.0, text="Did you bring the diary?"),
        SubtitleSegment(index=2, start_sec=175.0, end_sec=180.0, text="The diary is dangerous."),
        SubtitleSegment(index=3, start_sec=205.0, end_sec=210.0, text="Destroy the diary now!"),
    ]

    out = build_script_video_alignment(
        candidates_json,
        scene_inventory_json,
        subtitle_segments,
        search_radius_sec=90.0,
        default_window_sec=60.0,
        neighborhood=0,
    )

    assert out["meta"]["alignments_returned"] == 2

    first, second = out["alignments"]
    assert first["scene_idx_non_omitted"] == 2
    assert second["scene_idx_non_omitted"] == 10
    assert first["aligned_end_sec"] <= second["aligned_start_sec"]
    assert "diary" in first["matched_subtitle_text"].lower()
    assert "diary" in second["matched_subtitle_text"].lower()
    assert first["matched_subtitle_indices"] == [1]
    assert second["matched_subtitle_indices"] in ([2], [3])
    assert first["confidence"] > 0.5
    assert second["confidence"] > 0.5


def test_build_script_video_alignment_falls_back_to_expected_time_window_when_needed():
    candidates_json = {
        "candidates": [
            {
                "object_id": "locket",
                "canonical_name": "locket",
                "candidate_source": "inventory_only",
                "loose_scene_chain": [
                    {
                        "scene_idx_non_omitted": 6,
                        "heading": "INT. ATTIC - NIGHT",
                        "matched_inventory_objects": ["locket"],
                        "inventory_evidence": ["A locket rests in the attic."],
                    }
                ],
            }
        ]
    }
    scene_inventory_json = {
        "meta": {"scenes_non_omitted": 12},
        "scenes": {},
    }
    subtitle_segments = [
        SubtitleSegment(index=0, start_sec=0.0, end_sec=10.0, text="Hello there."),
        SubtitleSegment(index=1, start_sec=100.0, end_sec=110.0, text="Nothing relevant here."),
    ]

    out = build_script_video_alignment(
        candidates_json,
        scene_inventory_json,
        subtitle_segments,
        search_radius_sec=5.0,
        default_window_sec=30.0,
        neighborhood=0,
    )

    alignment = out["alignments"][0]
    assert alignment["matched_subtitle_indices"] == []
    assert alignment["matched_subtitle_text"] == ""
    assert alignment["confidence"] == 0.0
    assert alignment["aligned_end_sec"] > alignment["aligned_start_sec"]


def test_build_script_video_alignment_rejects_nearby_but_object_irrelevant_subtitles():
    candidates_json = {
        "candidates": [
            {
                "object_id": "fawkes",
                "canonical_name": "Fawkes",
                "candidate_source": "inventory_with_stage1_prior",
                "loose_scene_chain": [
                    {
                        "scene_idx_non_omitted": 9,
                        "heading": "INT. DUMBLEDORE'S OFFICE - DAY",
                        "matched_inventory_objects": ["Fawkes"],
                        "inventory_evidence": ["Fawkes perches on a pedestal."],
                    }
                ],
            }
        ]
    }
    scene_inventory_json = {
        "meta": {"scenes_non_omitted": 12},
        "scenes": {},
    }
    subtitle_segments = [
        SubtitleSegment(index=0, start_sec=150.0, end_sec=156.0, text="Hermione, welcome back. It is good to see you."),
        SubtitleSegment(index=1, start_sec=160.0, end_sec=166.0, text="Everyone is happy to be here today."),
    ]

    out = build_script_video_alignment(
        candidates_json,
        scene_inventory_json,
        subtitle_segments,
        search_radius_sec=90.0,
        default_window_sec=40.0,
        neighborhood=0,
    )

    alignment = out["alignments"][0]
    assert alignment["matched_subtitle_indices"] == []
    assert alignment["matched_subtitle_text"] == ""
    assert alignment["confidence"] == 0.0
