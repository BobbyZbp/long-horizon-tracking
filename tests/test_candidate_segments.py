"""
test_candidate_segments.py

Unit tests for candidate segment filtering.
"""
from __future__ import annotations

from tracking_project.pipeline.candidate_segments import build_candidate_segments


def test_build_candidate_segments_merges_nearby_windows_and_preserves_reasons():
    candidates_json = {
        "candidates": [
            {
                "object_id": "diary",
                "canonical_name": "Tom Riddle's diary",
                "candidate_source": "inventory_with_stage1_prior",
                "possible_chekhov_score": 0.81,
                "loose_scene_chain": [
                    {
                        "scene_idx_non_omitted": 3,
                        "role_tags": ["setup_candidate"],
                    },
                    {
                        "scene_idx_non_omitted": 4,
                        "role_tags": ["payoff_candidate", "high_climax_scene", "action_supported_match"],
                    },
                ],
            }
        ]
    }
    alignment_json = {
        "alignments": [
            {
                "candidate_index": 0,
                "object_id": "diary",
                "canonical_name": "Tom Riddle's diary",
                "candidate_source": "inventory_with_stage1_prior",
                "scene_idx_non_omitted": 3,
                "aligned_start_sec": 100.0,
                "aligned_end_sec": 110.0,
                "confidence": 0.61,
                "matched_subtitle_text": "I found the diary.",
                "matched_subtitle_indices": [10],
                "query_terms": ["diary"],
                "component_scores": {
                    "lexical_overlap": 0.4,
                    "object_overlap": 0.5,
                },
            },
            {
                "candidate_index": 0,
                "object_id": "diary",
                "canonical_name": "Tom Riddle's diary",
                "candidate_source": "inventory_with_stage1_prior",
                "scene_idx_non_omitted": 4,
                "aligned_start_sec": 113.0,
                "aligned_end_sec": 120.0,
                "confidence": 0.72,
                "matched_subtitle_text": "Destroy the diary!",
                "matched_subtitle_indices": [11],
                "query_terms": ["destroy", "diary"],
                "component_scores": {
                    "lexical_overlap": 0.7,
                    "object_overlap": 0.9,
                },
            },
        ]
    }

    out = build_candidate_segments(
        candidates_json,
        alignment_json,
        min_alignment_confidence=0.2,
        padding_sec=5.0,
        merge_gap_sec=4.0,
        max_segments_per_object=4,
    )

    assert len(out["segments"]) == 1
    seg = out["segments"][0]
    assert seg["object_id"] == "diary"
    assert seg["segment_role"] == "payoff"
    assert seg["source_scene_indices"] == [3, 4]
    assert seg["segment_start_sec"] == 95.0
    assert seg["segment_end_sec"] == 125.0
    assert "role:setup_candidate" in seg["reason_tags"]
    assert "role:payoff_candidate" in seg["reason_tags"]
    assert "subtitle_object_overlap" in seg["reason_tags"]
    assert len(seg["matched_subtitle_texts"]) == 2


def test_build_candidate_segments_caps_segments_per_object():
    candidates_json = {
        "candidates": [
            {
                "object_id": "locket",
                "canonical_name": "locket",
                "candidate_source": "inventory_only",
                "possible_chekhov_score": 0.66,
                "loose_scene_chain": [
                    {"scene_idx_non_omitted": 1, "role_tags": ["setup_candidate"]},
                    {"scene_idx_non_omitted": 5, "role_tags": []},
                    {"scene_idx_non_omitted": 9, "role_tags": ["payoff_candidate"]},
                ],
            }
        ]
    }
    alignment_json = {
        "alignments": [
            {
                "candidate_index": 0,
                "object_id": "locket",
                "canonical_name": "locket",
                "candidate_source": "inventory_only",
                "scene_idx_non_omitted": 1,
                "aligned_start_sec": 10.0,
                "aligned_end_sec": 14.0,
                "confidence": 0.3,
                "matched_subtitle_text": "",
                "matched_subtitle_indices": [],
                "query_terms": ["locket"],
                "component_scores": {},
            },
            {
                "candidate_index": 0,
                "object_id": "locket",
                "canonical_name": "locket",
                "candidate_source": "inventory_only",
                "scene_idx_non_omitted": 5,
                "aligned_start_sec": 60.0,
                "aligned_end_sec": 64.0,
                "confidence": 0.5,
                "matched_subtitle_text": "",
                "matched_subtitle_indices": [],
                "query_terms": ["locket"],
                "component_scores": {},
            },
            {
                "candidate_index": 0,
                "object_id": "locket",
                "canonical_name": "locket",
                "candidate_source": "inventory_only",
                "scene_idx_non_omitted": 9,
                "aligned_start_sec": 120.0,
                "aligned_end_sec": 124.0,
                "confidence": 0.8,
                "matched_subtitle_text": "",
                "matched_subtitle_indices": [],
                "query_terms": ["locket"],
                "component_scores": {},
            },
        ]
    }

    out = build_candidate_segments(
        candidates_json,
        alignment_json,
        max_segments_per_object=2,
    )

    assert len(out["segments"]) == 2
    assert out["segments"][0]["priority_score"] >= out["segments"][1]["priority_score"]


def test_build_candidate_segments_drops_lexical_only_subtitle_matches():
    candidates_json = {
        "candidates": [
            {
                "object_id": "camera",
                "canonical_name": "camera",
                "candidate_source": "inventory_only",
                "possible_chekhov_score": 0.72,
                "loose_scene_chain": [
                    {"scene_idx_non_omitted": 10, "role_tags": ["payoff_candidate"]},
                ],
            }
        ]
    }
    alignment_json = {
        "alignments": [
            {
                "candidate_index": 0,
                "object_id": "camera",
                "canonical_name": "camera",
                "candidate_source": "inventory_only",
                "scene_idx_non_omitted": 10,
                "aligned_start_sec": 100.0,
                "aligned_end_sec": 110.0,
                "confidence": 0.5,
                "matched_subtitle_text": "We'll be sending Potter to the hospital wing in a matchbox.",
                "matched_subtitle_indices": [50],
                "query_terms": ["camera", "hospital", "wing"],
                "component_scores": {
                    "lexical_overlap": 0.66,
                    "object_overlap": 0.0,
                },
            }
        ]
    }

    out = build_candidate_segments(
        candidates_json,
        alignment_json,
        min_alignment_confidence=0.15,
    )

    assert out["segments"] == []
    assert out["meta"]["skipped_non_object_specific_alignments"] == 1
