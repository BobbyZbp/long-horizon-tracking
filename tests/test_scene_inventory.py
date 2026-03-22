"""
test_scene_inventory.py

Unit tests for the high-recall double-channel scene inventory built before
cross-scene narrative validation and any video-facing alignment.
"""
from __future__ import annotations

from tracking_project.io.pdf_scene_parser import Scene
from tracking_project.pipeline.scene_inventory import build_scene_inventory


def test_build_scene_inventory_keeps_action_entities_even_when_tracking_keep_is_false(monkeypatch):
    scene = Scene(
        scene_idx=0,
        scene_idx_non_omitted=0,
        scene_label_raw="1",
        scene_label_int=1,
        marker_type="INT.",
        heading="INT. BEDROOM - NIGHT",
        page_start=1,
        page_end=1,
        lines=[
            "Harry sets the diary on the desk.",
            "",
            "GINNY",
            "The diary is back.",
        ],
    )

    def fake_parse_pdf_scenes(_pdf):
        return [scene], {"scene_markers_non_omitted": 1}

    def fake_action_extract(**kwargs):
        assert "Harry sets the diary on the desk." in kwargs["action_text"]
        return [
            {
                "canonical_name": "Tom Riddle's diary",
                "surface_forms": ["diary"],
                "category": "artifact",
                "physical_visualizable": True,
                "narrative_action_centrality": 0.71,
                "visual_grounded_score": 0.92,
                "state_change_potential": 0.86,
                "confidence": 0.78,
                "keep": False,
                "evidence": ["Harry sets the diary on the desk."],
            }
        ]

    def fake_dialogue_extract(**kwargs):
        assert kwargs["channel_name"] == "dialogue"
        return [
            {
                "canonical_name": "diary",
                "surface_forms": ["the diary"],
                "category": "artifact",
                "physical_visualizable": True,
                "direct_visual_evidence": False,
                "mention_strength": 0.42,
                 "confidence": 0.58,
                "evidence": ["The diary is back."],
            }
        ]

    monkeypatch.setattr("tracking_project.pipeline.scene_inventory.parse_pdf_scenes", fake_parse_pdf_scenes)
    monkeypatch.setattr(
        "tracking_project.pipeline.scene_inventory.openai_scene_extract_entities",
        fake_action_extract,
    )
    monkeypatch.setattr(
        "tracking_project.pipeline.scene_inventory.openai_scene_inventory_extract_entities",
        fake_dialogue_extract,
    )

    out = build_scene_inventory(
        pdf="dummy.pdf",
        llm_model="fake-model",
        cache_dir="dummy-cache",
        dialogue_mode="llm",
    )

    scene0 = out["scenes"]["0"]
    assert len(scene0["action_objects"]) == 1
    assert scene0["action_objects"][0]["tracking_keep"] is False

    assert len(scene0["merged_loose_objects"]) == 1
    merged = scene0["merged_loose_objects"][0]
    assert merged["canonical_name"] == "Tom Riddle's diary"
    assert merged["source_channels"] == ["action", "dialogue"]
    assert "diary" in merged["normalized_aliases"]
    assert "tom riddle's diary" in merged["normalized_aliases"]


def test_build_scene_inventory_hybrid_dialogue_mode_falls_back_to_heuristics(monkeypatch):
    scene = Scene(
        scene_idx=0,
        scene_idx_non_omitted=0,
        scene_label_raw="1",
        scene_label_int=1,
        marker_type="INT.",
        heading="INT. HALL - DAY",
        page_start=1,
        page_end=1,
        lines=[
            "RON",
            "Bring the key.",
            "",
            "HERMIONE",
            "Bring the sword.",
        ],
    )

    monkeypatch.setattr(
        "tracking_project.pipeline.scene_inventory.parse_pdf_scenes",
        lambda _pdf: ([scene], {"scene_markers_non_omitted": 1}),
    )
    monkeypatch.setattr(
        "tracking_project.pipeline.scene_inventory.openai_scene_extract_entities",
        lambda **_kwargs: [],
    )

    def fake_dialogue_extract(**_kwargs):
        raise RuntimeError("LLM unavailable")

    monkeypatch.setattr(
        "tracking_project.pipeline.scene_inventory.openai_scene_inventory_extract_entities",
        fake_dialogue_extract,
    )

    out = build_scene_inventory(
        pdf="dummy.pdf",
        llm_model="fake-model",
        cache_dir="dummy-cache",
        dialogue_mode="hybrid",
    )

    scene0 = out["scenes"]["0"]
    names = {rec["canonical_name"].lower() for rec in scene0["dialogue_objects"]}
    assert names == {"key", "sword"}
    assert {rec["canonical_name"].lower() for rec in scene0["merged_loose_objects"]} == {"key", "sword"}
