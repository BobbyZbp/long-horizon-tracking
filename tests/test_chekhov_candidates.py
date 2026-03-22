"""
test_chekhov_candidates.py

Unit tests for scene-level loose candidate mining and cross-scene narrative
validation before any video-facing alignment.
"""
from __future__ import annotations

from tracking_project.pipeline.chekhov_candidates import build_chekhov_candidates


def test_build_chekhov_candidates_uses_scene_inventory_with_stage1_prior():
    events_json = {
        "meta": {"gap_quantiles": {"q50": 4.0, "q75": 6.0}},
        "events": [
            {
                "entity_key": "tom riddles diary",
                "canonical_name": "Tom Riddle's diary",
                "entity_type": "artifact",
                "surface_forms": ["diary"],
                "scene_a_idx_non_omitted": 2,
                "scene_b_idx_non_omitted": 10,
                "gap_jump": 8,
                "gap_absent": 7,
                "freq_scenes": 2,
                "coverage": 0.22,
                "scores": {
                    "narrative_action_centrality": 0.88,
                    "visual_grounded_score": 0.82,
                    "state_change_potential": 0.74,
                    "confidence": 0.81,
                },
                "provenance": {"scenes_non_omitted_list": [2, 10]},
            }
        ],
    }
    scene_inventory_json = {
        "meta": {"scenes_non_omitted": 12},
        "scenes": {
            "2": {
                "heading": "INT. ROOM - DAY",
                "merged_loose_objects": [
                    {
                        "object_id": "tom riddles diary",
                        "canonical_name": "Tom Riddle's diary",
                        "surface_forms": ["diary"],
                        "normalized_aliases": ["tom riddles diary", "diary"],
                        "source_channels": ["action"],
                        "category": "artifact",
                        "direct_visual_evidence": True,
                        "mention_strength_max": 0.8,
                        "confidence_max": 0.85,
                        "evidence": ["Harry opens the diary."],
                    }
                ],
            },
            "9": {
                "heading": "INT. LIBRARY - NIGHT",
                "merged_loose_objects": [
                    {
                        "object_id": "diary",
                        "canonical_name": "diary",
                        "surface_forms": ["the diary"],
                        "normalized_aliases": ["diary"],
                        "source_channels": ["dialogue"],
                        "category": "artifact",
                        "direct_visual_evidence": False,
                        "mention_strength_max": 0.42,
                        "confidence_max": 0.6,
                        "evidence": ["Ginny mentions the diary."],
                    }
                ],
            },
            "10": {
                "heading": "INT. CHAMBER - NIGHT",
                "merged_loose_objects": [
                    {
                        "object_id": "tom riddles diary",
                        "canonical_name": "Tom Riddle's diary",
                        "surface_forms": ["diary"],
                        "normalized_aliases": ["tom riddles diary", "diary"],
                        "source_channels": ["action"],
                        "category": "artifact",
                        "direct_visual_evidence": True,
                        "mention_strength_max": 0.93,
                        "confidence_max": 0.88,
                        "evidence": ["Harry stabs the diary."],
                    }
                ],
            },
        },
    }
    scene_climax_json = {
        "meta": {"scenes_non_omitted": 12},
        "scenes": {
            "2": {"heading": "INT. ROOM - DAY", "climax_score": 15},
            "9": {"heading": "INT. LIBRARY - NIGHT", "climax_score": 82},
            "10": {"heading": "INT. CHAMBER - NIGHT", "climax_score": 91},
        },
    }

    out = build_chekhov_candidates(
        events_json,
        scene_inventory_json,
        scene_climax_json,
        min_validation_score=0.55,
    )

    assert len(out["candidates"]) == 1
    candidate = out["candidates"][0]

    assert candidate["candidate_source"] == "inventory_with_stage1_prior"
    assert candidate["possible_chekhov_label"] == "high"
    assert candidate["possible_chekhov_score"] >= 0.72
    assert "scene_inventory_loose_reinforcement" in candidate["validation_reasons"]
    assert "stage1_prior_support" in candidate["validation_reasons"]
    assert candidate["stage1_prior"]["source_event_index"] == 0

    chain_scene_indices = [rec["scene_idx_non_omitted"] for rec in candidate["loose_scene_chain"]]
    assert chain_scene_indices == [2, 9, 10]
    assert candidate["loose_scene_chain"][1]["matched_source_channels"] == ["dialogue"]
    assert candidate["loose_scene_chain"][2]["matched_source_channels"] == ["action"]
    assert candidate["setup_scene_indices"] == [2]
    assert candidate["payoff_scene_indices"] == [10]


def test_build_chekhov_candidates_keeps_inventory_only_object_when_pattern_is_strong():
    scene_inventory_json = {
        "meta": {"scenes_non_omitted": 12},
        "scenes": {
            "1": {
                "heading": "INT. STUDY - DAY",
                "merged_loose_objects": [
                    {
                        "object_id": "locket",
                        "canonical_name": "silver locket",
                        "surface_forms": ["locket"],
                        "normalized_aliases": ["silver locket", "locket"],
                        "source_channels": ["action"],
                        "category": "artifact",
                        "direct_visual_evidence": True,
                        "mention_strength_max": 0.77,
                        "confidence_max": 0.8,
                        "evidence": ["A silver locket sits on the desk."],
                    }
                ],
            },
            "8": {
                "heading": "INT. ATTIC - NIGHT",
                "merged_loose_objects": [
                    {
                        "object_id": "locket",
                        "canonical_name": "locket",
                        "surface_forms": ["the locket"],
                        "normalized_aliases": ["locket"],
                        "source_channels": ["action", "dialogue"],
                        "category": "artifact",
                        "direct_visual_evidence": True,
                        "mention_strength_max": 0.9,
                        "confidence_max": 0.84,
                        "evidence": ["She opens the locket."],
                    }
                ],
            },
        },
    }
    scene_climax_json = {
        "meta": {"scenes_non_omitted": 12},
        "scenes": {
            "1": {"heading": "INT. STUDY - DAY", "climax_score": 10},
            "8": {"heading": "INT. ATTIC - NIGHT", "climax_score": 86},
        },
    }

    out = build_chekhov_candidates(
        {"meta": {"gap_quantiles": {"q50": 4.0, "q75": 6.0}}, "events": []},
        scene_inventory_json,
        scene_climax_json,
        min_validation_score=0.55,
    )

    assert len(out["candidates"]) == 1
    candidate = out["candidates"][0]

    assert candidate["canonical_name"] == "silver locket"
    assert candidate["candidate_source"] == "inventory_only"
    assert candidate["stage1_prior"] is None
    assert candidate["inventory_support"]["inventory_scene_count"] == 2
    assert candidate["possible_chekhov_score"] >= 0.55
    assert "inventory_only_candidate" in candidate["validation_reasons"]


def test_build_chekhov_candidates_filters_low_validation_inventory_objects():
    scene_inventory_json = {
        "meta": {"scenes_non_omitted": 12},
        "scenes": {
            "6": {
                "heading": "INT. ROOM - DAY",
                "merged_loose_objects": [
                    {
                        "object_id": "chair",
                        "canonical_name": "chair",
                        "surface_forms": ["chair"],
                        "normalized_aliases": ["chair"],
                        "source_channels": ["action"],
                        "category": "object",
                        "direct_visual_evidence": True,
                        "mention_strength_max": 0.14,
                        "confidence_max": 0.3,
                        "evidence": ["A chair stands by the wall."],
                    }
                ],
            },
            "7": {
                "heading": "INT. ROOM - DAY",
                "merged_loose_objects": [
                    {
                        "object_id": "chair",
                        "canonical_name": "chair",
                        "surface_forms": ["chair"],
                        "normalized_aliases": ["chair"],
                        "source_channels": ["action"],
                        "category": "object",
                        "direct_visual_evidence": True,
                        "mention_strength_max": 0.12,
                        "confidence_max": 0.25,
                        "evidence": ["The same chair remains there."],
                    }
                ],
            },
        },
    }
    scene_climax_json = {
        "meta": {"scenes_non_omitted": 12},
        "scenes": {
            "6": {"heading": "INT. ROOM - DAY", "climax_score": 10},
            "7": {"heading": "INT. ROOM - DAY", "climax_score": 12},
        },
    }

    out = build_chekhov_candidates(
        {"meta": {"gap_quantiles": {"q50": 4.0, "q75": 6.0}}, "events": []},
        scene_inventory_json,
        scene_climax_json,
        min_validation_score=0.55,
    )

    assert out["candidates"] == []


def test_build_chekhov_candidates_prefers_specific_stage1_name_over_generic_inventory_name():
    events_json = {
        "meta": {"gap_quantiles": {"q50": 4.0, "q75": 6.0}},
        "events": [
            {
                "entity_key": "fawkes",
                "canonical_name": "Fawkes",
                "entity_type": "creature",
                "surface_forms": ["bird", "phoenix"],
                "scene_a_idx_non_omitted": 3,
                "scene_b_idx_non_omitted": 9,
                "gap_jump": 6,
                "gap_absent": 5,
                "freq_scenes": 2,
                "coverage": 0.22,
                "scores": {
                    "narrative_action_centrality": 0.8,
                    "visual_grounded_score": 0.9,
                    "state_change_potential": 0.5,
                    "confidence": 0.9,
                },
                "provenance": {"scenes_non_omitted_list": [3, 9]},
            }
        ],
    }
    scene_inventory_json = {
        "meta": {"scenes_non_omitted": 12},
        "scenes": {
            "3": {
                "heading": "INT. CLASSROOM - DAY",
                "merged_loose_objects": [
                    {
                        "object_id": "animal",
                        "canonical_name": "animal",
                        "surface_forms": ["bird"],
                        "normalized_aliases": ["animal", "bird"],
                        "source_channels": ["action"],
                        "category": "creature",
                        "direct_visual_evidence": True,
                        "mention_strength_max": 0.8,
                        "confidence_max": 0.8,
                        "evidence": ["A bird sits on the desk."],
                    }
                ],
            },
            "9": {
                "heading": "INT. OFFICE - NIGHT",
                "merged_loose_objects": [
                    {
                        "object_id": "fawkes",
                        "canonical_name": "Fawkes",
                        "surface_forms": ["phoenix"],
                        "normalized_aliases": ["fawkes", "phoenix"],
                        "source_channels": ["action"],
                        "category": "creature",
                        "direct_visual_evidence": True,
                        "mention_strength_max": 0.9,
                        "confidence_max": 0.9,
                        "evidence": ["Fawkes lands beside Harry."],
                    }
                ],
            },
        },
    }
    scene_climax_json = {
        "meta": {"scenes_non_omitted": 12},
        "scenes": {
            "3": {"heading": "INT. CLASSROOM - DAY", "climax_score": 20},
            "9": {"heading": "INT. OFFICE - NIGHT", "climax_score": 90},
        },
    }

    out = build_chekhov_candidates(events_json, scene_inventory_json, scene_climax_json, min_validation_score=0.55)

    assert len(out["candidates"]) == 1
    candidate = out["candidates"][0]
    assert candidate["object_id"] == "fawkes"
    assert candidate["canonical_name"] == "Fawkes"
    assert candidate["stage1_prior"]["canonical_name"] == "Fawkes"


def test_build_chekhov_candidates_rejects_generic_only_stage1_alias_hijack_and_dedups_same_object():
    events_json = {
        "meta": {"gap_quantiles": {"q50": 4.0, "q75": 6.0}},
        "events": [
            {
                "entity_key": "flying ford anglia",
                "canonical_name": "Ford Anglia",
                "entity_type": "vehicle",
                "surface_forms": ["car", "ford anglia"],
                "scene_a_idx_non_omitted": 2,
                "scene_b_idx_non_omitted": 8,
                "gap_jump": 6,
                "gap_absent": 5,
                "freq_scenes": 2,
                "coverage": 0.2,
                "scores": {
                    "narrative_action_centrality": 0.9,
                    "visual_grounded_score": 0.95,
                    "state_change_potential": 0.6,
                    "confidence": 0.9,
                },
                "provenance": {"scenes_non_omitted_list": [2, 8]},
            },
            {
                "entity_key": "flying ford anglia",
                "canonical_name": "Flying Ford Anglia",
                "entity_type": "vehicle",
                "surface_forms": ["anglia", "car"],
                "scene_a_idx_non_omitted": 3,
                "scene_b_idx_non_omitted": 9,
                "gap_jump": 6,
                "gap_absent": 5,
                "freq_scenes": 2,
                "coverage": 0.2,
                "scores": {
                    "narrative_action_centrality": 0.85,
                    "visual_grounded_score": 0.92,
                    "state_change_potential": 0.58,
                    "confidence": 0.88,
                },
                "provenance": {"scenes_non_omitted_list": [3, 9]},
            },
            {
                "entity_key": "diary",
                "canonical_name": "diary",
                "entity_type": "artifact",
                "surface_forms": ["book"],
                "scene_a_idx_non_omitted": 6,
                "scene_b_idx_non_omitted": 10,
                "gap_jump": 4,
                "gap_absent": 3,
                "freq_scenes": 2,
                "coverage": 0.18,
                "scores": {
                    "narrative_action_centrality": 0.82,
                    "visual_grounded_score": 0.86,
                    "state_change_potential": 0.55,
                    "confidence": 0.84,
                },
                "provenance": {"scenes_non_omitted_list": [6, 10]},
            },
        ],
    }
    scene_inventory_json = {
        "meta": {"scenes_non_omitted": 12},
        "scenes": {
            "2": {
                "heading": "EXT. STREET - DAY",
                "merged_loose_objects": [
                    {
                        "object_id": "flying ford anglia",
                        "canonical_name": "Flying Ford Anglia",
                        "surface_forms": ["car"],
                        "normalized_aliases": ["flying ford anglia", "ford anglia", "car"],
                        "source_channels": ["action"],
                        "category": "vehicle",
                        "direct_visual_evidence": True,
                        "mention_strength_max": 0.9,
                        "confidence_max": 0.9,
                        "evidence": ["The Flying Ford Anglia lands in the street."],
                    }
                ],
            },
            "8": {
                "heading": "EXT. FOREST - NIGHT",
                "merged_loose_objects": [
                    {
                        "object_id": "flying ford anglia",
                        "canonical_name": "Flying Ford Anglia",
                        "surface_forms": ["anglia"],
                        "normalized_aliases": ["flying ford anglia", "ford anglia", "anglia"],
                        "source_channels": ["action"],
                        "category": "vehicle",
                        "direct_visual_evidence": True,
                        "mention_strength_max": 0.95,
                        "confidence_max": 0.9,
                        "evidence": ["The Anglia crashes through the trees."],
                    }
                ],
            },
            "1": {
                "heading": "INT. BOOKSHOP - DAY",
                "merged_loose_objects": [
                    {
                        "object_id": "moste potente potions",
                        "canonical_name": "Moste Potente Potions",
                        "surface_forms": ["potions"],
                        "normalized_aliases": ["moste potente potions", "potions"],
                        "source_channels": ["action"],
                        "category": "artifact",
                        "direct_visual_evidence": True,
                        "mention_strength_max": 0.88,
                        "confidence_max": 0.85,
                        "evidence": ["Hermione reaches for Moste Potente Potions."],
                    }
                ],
            },
            "10": {
                "heading": "INT. LIBRARY - NIGHT",
                "merged_loose_objects": [
                    {
                        "object_id": "moste potente potions",
                        "canonical_name": "Moste Potente Potions",
                        "surface_forms": ["the potions book"],
                        "normalized_aliases": ["moste potente potions", "potions book"],
                        "source_channels": ["action"],
                        "category": "artifact",
                        "direct_visual_evidence": True,
                        "mention_strength_max": 0.9,
                        "confidence_max": 0.86,
                        "evidence": ["The Moste Potente Potions text lies open on the desk."],
                    }
                ],
            },
        },
    }
    scene_climax_json = {
        "meta": {"scenes_non_omitted": 12},
        "scenes": {
            "1": {"heading": "INT. BOOKSHOP - DAY", "climax_score": 15},
            "2": {"heading": "EXT. STREET - DAY", "climax_score": 22},
            "8": {"heading": "EXT. FOREST - NIGHT", "climax_score": 88},
            "10": {"heading": "INT. LIBRARY - NIGHT", "climax_score": 82},
        },
    }

    out = build_chekhov_candidates(events_json, scene_inventory_json, scene_climax_json, min_validation_score=0.55)

    assert len([c for c in out["candidates"] if c["object_id"] == "flying ford anglia"]) == 1
    anglia = next(c for c in out["candidates"] if c["object_id"] == "flying ford anglia")
    assert len(anglia["stage1_supports"]) == 2

    potions = next(c for c in out["candidates"] if c["object_id"] == "moste potente potions")
    assert potions["candidate_source"] == "inventory_only"
    assert potions["stage1_prior"] is None


def test_build_chekhov_candidates_does_not_merge_inventory_groups_on_generic_book_alias():
    scene_inventory_json = {
        "meta": {"scenes_non_omitted": 12},
        "scenes": {
            "1": {
                "heading": "INT. LIBRARY - DAY",
                "merged_loose_objects": [
                    {
                        "object_id": "moste potente potions",
                        "canonical_name": "Moste Potente Potions",
                        "surface_forms": ["book"],
                        "normalized_aliases": ["moste potente potions", "book"],
                        "source_channels": ["action"],
                        "category": "artifact",
                        "direct_visual_evidence": True,
                        "mention_strength_max": 0.9,
                        "confidence_max": 0.9,
                        "evidence": ["Hermione reaches for Moste Potente Potions."],
                    }
                ],
            },
            "6": {
                "heading": "INT. CORRIDOR - NIGHT",
                "merged_loose_objects": [
                    {
                        "object_id": "diary",
                        "canonical_name": "diary",
                        "surface_forms": ["book"],
                        "normalized_aliases": ["diary", "book"],
                        "source_channels": ["action"],
                        "category": "artifact",
                        "direct_visual_evidence": True,
                        "mention_strength_max": 0.9,
                        "confidence_max": 0.9,
                        "evidence": ["Harry opens the diary."],
                    }
                ],
            },
            "10": {
                "heading": "INT. LIBRARY - NIGHT",
                "merged_loose_objects": [
                    {
                        "object_id": "moste potente potions",
                        "canonical_name": "Moste Potente Potions",
                        "surface_forms": ["potions book"],
                        "normalized_aliases": ["moste potente potions", "potions book"],
                        "source_channels": ["action"],
                        "category": "artifact",
                        "direct_visual_evidence": True,
                        "mention_strength_max": 0.9,
                        "confidence_max": 0.9,
                        "evidence": ["The Moste Potente Potions text lies open."],
                    }
                ],
            },
            "11": {
                "heading": "INT. CHAMBER - NIGHT",
                "merged_loose_objects": [
                    {
                        "object_id": "diary",
                        "canonical_name": "diary",
                        "surface_forms": ["the diary"],
                        "normalized_aliases": ["diary"],
                        "source_channels": ["action"],
                        "category": "artifact",
                        "direct_visual_evidence": True,
                        "mention_strength_max": 0.9,
                        "confidence_max": 0.9,
                        "evidence": ["Harry destroys the diary."],
                    }
                ],
            },
        },
    }
    scene_climax_json = {
        "meta": {"scenes_non_omitted": 12},
        "scenes": {
            "1": {"heading": "INT. LIBRARY - DAY", "climax_score": 18},
            "6": {"heading": "INT. CORRIDOR - NIGHT", "climax_score": 55},
            "10": {"heading": "INT. LIBRARY - NIGHT", "climax_score": 70},
            "11": {"heading": "INT. CHAMBER - NIGHT", "climax_score": 95},
        },
    }

    out = build_chekhov_candidates(
        {"meta": {"gap_quantiles": {"q50": 4.0, "q75": 6.0}}, "events": []},
        scene_inventory_json,
        scene_climax_json,
        min_validation_score=0.55,
    )

    object_ids = {candidate["object_id"] for candidate in out["candidates"]}
    assert "moste potente potions" in object_ids
    assert "diary" in object_ids

    potions = next(c for c in out["candidates"] if c["object_id"] == "moste potente potions")
    diary = next(c for c in out["candidates"] if c["object_id"] == "diary")

    assert potions["inventory_support"]["inventory_scene_indices"] == [1, 10]
    assert diary["inventory_support"]["inventory_scene_indices"] == [6, 11]
