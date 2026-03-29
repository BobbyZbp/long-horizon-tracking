"""
test_deliberate_resampling.py

Unit tests for deliberate object-centric resampling helpers.
"""
from __future__ import annotations

from tracking_project.pipeline.deliberate_resampling import (
    build_explicit_window,
    build_centered_window,
    select_segment,
)


def test_select_segment_prefers_highest_priority_for_object():
    segments_json = {
        "segments": [
            {
                "segment_id": "seg_low",
                "object_id": "sorting hat",
                "canonical_name": "Sorting Hat",
                "segment_role": "setup",
                "priority_score": 0.5,
                "alignment_confidence_max": 0.9,
                "segment_start_sec": 10.0,
                "segment_end_sec": 20.0,
                "source_scene_indices": [10],
            },
            {
                "segment_id": "seg_high",
                "object_id": "sorting hat",
                "canonical_name": "Sorting Hat",
                "segment_role": "payoff",
                "priority_score": 0.9,
                "alignment_confidence_max": 0.7,
                "segment_start_sec": 30.0,
                "segment_end_sec": 40.0,
                "source_scene_indices": [20],
            },
        ]
    }

    selected = select_segment(segments_json, object_id="sorting hat")
    assert selected.segment_id == "seg_high"
    assert selected.segment_role == "payoff"


def test_build_centered_window_expands_around_midpoint_and_clamps():
    segments_json = {
        "segments": [
            {
                "segment_id": "seg_a",
                "object_id": "fawkes",
                "canonical_name": "Fawkes",
                "segment_role": "payoff",
                "priority_score": 1.0,
                "alignment_confidence_max": 0.8,
                "segment_start_sec": 100.0,
                "segment_end_sec": 140.0,
                "source_scene_indices": [108],
            }
        ]
    }

    selected = select_segment(segments_json, object_id="fawkes")
    window = build_centered_window(
        selected,
        window_radius_sec=120.0,
        video_duration_sec=500.0,
    )

    assert window["center_sec"] == 120.0
    assert window["window_start_sec"] == 0.0
    assert window["window_end_sec"] == 240.0
    assert window["window_start_hhmmss"] == "00:00:00"
    assert window["window_end_hhmmss"] == "00:04:00"


def test_build_centered_window_uses_manual_center_when_provided():
    segments_json = {
        "segments": [
            {
                "segment_id": "seg_hat",
                "object_id": "sorting hat",
                "canonical_name": "Sorting Hat",
                "segment_role": "payoff",
                "priority_score": 1.0,
                "alignment_confidence_max": 0.8,
                "segment_start_sec": 8420.0,
                "segment_end_sec": 8456.0,
                "source_scene_indices": [108, 110],
            }
        ]
    }

    selected = select_segment(segments_json, object_id="sorting hat")
    window = build_centered_window(
        selected,
        window_radius_sec=120.0,
        center_sec=4980.0,
        video_duration_sec=10000.0,
    )

    assert window["center_sec"] == 4980.0
    assert window["window_start_sec"] == 4860.0
    assert window["window_end_sec"] == 5100.0
    assert window["window_start_hhmmss"] == "01:21:00"
    assert window["window_end_hhmmss"] == "01:25:00"


def test_build_explicit_window_preserves_manual_bounds():
    window = build_explicit_window(
        start_sec=4976.0,
        end_sec=4987.0,
        video_duration_sec=10000.0,
    )

    assert window["center_sec"] == 4981.5
    assert window["window_start_sec"] == 4976.0
    assert window["window_end_sec"] == 4987.0
    assert window["window_start_hhmmss"] == "01:22:56"
    assert window["window_end_hhmmss"] == "01:23:07"
