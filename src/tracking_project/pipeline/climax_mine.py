from __future__ import annotations

import json
from typing import Any, Dict, List

from tracking_project.io.pdf_scene_parser import parse_pdf_scenes
from tracking_project.llm.prompts_scene import build_climax_scoring_prompt
from tracking_project.llm.scene_extractor_openai import (
    call_llm_json,
    openai_scene_extract_entities,
)
from tracking_project.text.cleaners import clean_scene_lines
from tracking_project.text.blocks import split_blocks, block_type


def _scene_action_text(scene) -> str:
    """
    Build action_text exactly the same way as Stage 1, so object extraction
    behavior stays aligned with the main pipeline.
    """
    clean_lines = clean_scene_lines(scene.lines)
    blocks = split_blocks(clean_lines)

    action_chunks: List[str] = []
    for blk in blocks:
        if block_type(blk) == "action":
            txt = " ".join(ln.strip() for ln in blk if ln.strip())
            if txt:
                action_chunks.append(txt)

    return "\n".join(action_chunks).strip()


def build_scene_object_map(
    scenes,
    *,
    llm_model: str,
    cache_root: str = "data/llm_cache",
    keep_categories: List[str] | None = None,
) -> Dict[int, List[str]]:
    """
    Build a scene -> objects map from per-scene extraction, not from final
    curated run1.json events.

    This fixes the main bug: run1.json only contains filtered long-gap events,
    so it is not a complete object inventory for each scene.
    """
    keep_categories = keep_categories or [
        "object",
        "artifact",
        "weapon",
        "creature",
        "vehicle",
    ]

    scene_objects: Dict[int, List[str]] = {}

    for scene in scenes:
        scene_idx = scene.scene_idx_non_omitted
        action_text = _scene_action_text(scene)

        if not action_text:
            scene_objects[scene_idx] = []
            continue

        entities = openai_scene_extract_entities(
            model=llm_model,
            scene_idx_non_omitted=scene_idx,
            heading=scene.heading,
            action_text=action_text,
            cache_root=cache_root,
        )

        names: List[str] = []
        seen = set()

        for e in entities:
            if not e.get("keep", False):
                continue
            if not e.get("physical_visualizable", False):
                continue

            category = e.get("category", "other")
            if category not in keep_categories:
                continue

            name = str(e.get("canonical_name", "")).strip()
            if not name:
                continue

            if name not in seen:
                seen.add(name)
                names.append(name)

        scene_objects[scene_idx] = names

    return scene_objects


def mine_climax_llm(
    pdf: str,
    events: str,
    out: str,
    llm_model: str,
    cache_dir: str = "data/llm_cache",
) -> Dict[str, Any]:
    """
    Scene-level climax scoring with scene-level object attachment.

    Important fix:
    objects are now taken from per-scene extraction, not from the filtered
    Stage-1 events JSON.
    """
    scenes, scene_meta = parse_pdf_scenes(pdf)

    # Keep only non-omitted scenes
    scenes = [s for s in scenes if s.scene_idx_non_omitted >= 0]

    # Keep loading events for compatibility / future use,
    # but do not use them for scene-object pulling.
    with open(events, "r", encoding="utf-8") as f:
        events_json = json.load(f)

    scene_objects = build_scene_object_map(
        scenes,
        llm_model=llm_model,
        cache_root=cache_dir,
    )

    # Still placeholder global summary for now; not today's debug target.
    story_summary = (
        "Harry Potter returns to Hogwarts where mysterious attacks petrify students. "
        "The Chamber of Secrets is reopened and Harry eventually confronts Tom Riddle "
        "and the Basilisk."
    )

    result: Dict[str, Any] = {
        "meta": {
            "pdf": pdf,
            "events_source": events,
            "scenes_non_omitted": len(scenes),
            "object_source": "scene_level_extraction",
            "llm_model": llm_model,
        },
        "scenes": {},
    }

    for scene in scenes:
        action_text = _scene_action_text(scene)

        prompt = build_climax_scoring_prompt(
            story_summary,
            scene.scene_idx_non_omitted,
            scene.heading,
            action_text,
        )

        response = call_llm_json(prompt, model=llm_model)
        score = int(response["climax_score"])

        result["scenes"][str(scene.scene_idx_non_omitted)] = {
            "heading": scene.heading,
            "climax_score": score,
            "objects": scene_objects.get(scene.scene_idx_non_omitted, []),
        }

        print(
            f"Scene {scene.scene_idx_non_omitted} -> "
            f"climax={score}, objects={len(scene_objects.get(scene.scene_idx_non_omitted, []))}"
        )

    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result
