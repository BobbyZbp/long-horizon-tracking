"""
scene_inventory.py

This module builds a high-recall per-scene object inventory before any
cross-scene Chekhov validation or video alignment work begins.

Why this layer exists
---------------------
The current Stage-1 pipeline is already a filter:
    - it focuses on long-gap re-occurrence events
    - it uses the `keep` field for long-term tracking usefulness
    - it only consumes action text

That makes Stage-1 valuable as a precision-oriented ranking stage, but unsafe
as the sole source of candidate objects for Chekhov-style setup/payoff mining.

To avoid filtering on top of an already filtered source, this module builds a
separate scene-level inventory with higher recall.

Double-channel design
---------------------
Each scene is processed through two channels:

1) action_objects
   - uses the existing scene extractor
   - drops the Stage-1 `keep` gate
   - preserves all physical objects in allowed categories

2) dialogue_objects
   - can use a dedicated higher-recall LLM extractor
   - or a deterministic heuristic fallback when LLM access is unavailable

3) merged_loose_objects
   - merges both channels into a single high-recall scene inventory
   - records source channels so downstream code can distinguish stronger
     action evidence from weaker dialogue-only mentions
   - preserves normalized aliases so downstream cross-scene aggregation can
     merge "Tom Riddle's diary" with "the diary" instead of treating them as
     unrelated objects

This output is the correct upstream input for loose candidate mining and
cross-scene narrative validation.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence

from tracking_project.io.pdf_scene_parser import parse_pdf_scenes
from tracking_project.llm.scene_extractor_openai import (
    openai_scene_extract_entities,
    openai_scene_inventory_extract_entities,
)
from tracking_project.text.blocks import block_type, split_blocks
from tracking_project.text.cleaners import clean_scene_lines


_DIALOGUE_ARTICLE_RE = re.compile(
    r"\b(?:a|an|the|my|his|her|their|this|that|these|those|our|your)\s+"
    r"([A-Za-z][A-Za-z' -]{1,48})"
)
_DIALOGUE_TITLE_RE = re.compile(
    r"\b([A-Z][a-z']+(?:\s+[A-Z][a-z']+){1,3})\b"
)
_GENERIC_BLACKLIST = {
    "bit",
    "business",
    "day",
    "hand",
    "hour",
    "idea",
    "night",
    "one",
    "other hand",
    "place",
    "school",
    "something",
    "sound",
    "thing",
    "time",
    "voice",
    "way",
}


def _normalize_entity_key(name: str) -> str:
    text = name.strip().lower()
    text = re.sub(r"^(the|a|an|my|his|her|their|this|that|these|those|our|your)\s+", "", text)
    text = re.sub(r"[^\w\s']+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _record_aliases(rec: Dict[str, Any]) -> List[str]:
    aliases = []
    for raw in [
        rec.get("object_id", ""),
        rec.get("canonical_name", ""),
        *rec.get("surface_forms", []),
    ]:
        norm = _normalize_entity_key(str(raw))
        if norm and norm not in aliases:
            aliases.append(norm)
    return aliases


def _scene_channel_texts(scene) -> Dict[str, str]:
    clean_lines = clean_scene_lines(scene.lines)
    blocks = split_blocks(clean_lines)

    action_chunks: List[str] = []
    dialogue_chunks: List[str] = []

    for blk in blocks:
        if not blk:
            continue

        if block_type(blk) == "dialogue":
            spoken = " ".join(ln.strip() for ln in blk[1:] if ln.strip())
            if spoken:
                dialogue_chunks.append(spoken)
            continue

        action = " ".join(ln.strip() for ln in blk if ln.strip())
        if action:
            action_chunks.append(action)

    return {
        "action": "\n".join(action_chunks).strip(),
        "dialogue": "\n".join(dialogue_chunks).strip(),
    }


def _action_inventory_records(
    scene,
    *,
    llm_model: str,
    cache_root: str,
    keep_categories: Sequence[str],
) -> List[Dict[str, Any]]:
    channel_texts = _scene_channel_texts(scene)
    action_text = channel_texts["action"]
    if not action_text:
        return []

    entities = openai_scene_extract_entities(
        model=llm_model,
        scene_idx_non_omitted=scene.scene_idx_non_omitted,
        heading=scene.heading,
        action_text=action_text,
        cache_root=cache_root,
    )

    out: List[Dict[str, Any]] = []
    for ent in entities:
        if not ent.get("physical_visualizable", False):
            continue

        category = str(ent.get("category", "other"))
        if category not in keep_categories:
            continue

        canonical_name = str(ent.get("canonical_name", "")).strip()
        object_id = _normalize_entity_key(canonical_name)
        if not object_id:
            continue

        mention_strength = max(
            float(ent.get("narrative_action_centrality", 0.0)),
            float(ent.get("visual_grounded_score", 0.0)),
            float(ent.get("state_change_potential", 0.0)),
        )

        out.append(
            {
                "object_id": object_id,
                "canonical_name": canonical_name,
                "surface_forms": list(ent.get("surface_forms", [])),
                "category": category,
                "physical_visualizable": True,
                "direct_visual_evidence": True,
                "mention_strength": round(mention_strength, 4),
                "confidence": round(float(ent.get("confidence", 0.0)), 4),
                "evidence": list(ent.get("evidence", []))[:3],
                "source_channel": "action",
                "tracking_keep": bool(ent.get("keep", False)),
            }
        )

    return out


def _heuristic_dialogue_entities(dialogue_text: str) -> List[Dict[str, Any]]:
    """
    Deterministic high-recall fallback for dialogue-only object mentions.

    The goal here is not precision. It is to avoid returning an empty dialogue
    channel when LLM access is unavailable.
    """
    if not dialogue_text.strip():
        return []

    sentences = [
        sent.strip()
        for sent in re.split(r"(?<=[.!?])\s+", dialogue_text)
        if sent.strip()
    ]

    found: Dict[str, Dict[str, Any]] = {}
    for sent in sentences:
        candidates = []
        candidates.extend(match.group(1).strip() for match in _DIALOGUE_ARTICLE_RE.finditer(sent))
        candidates.extend(match.group(1).strip() for match in _DIALOGUE_TITLE_RE.finditer(sent))

        for raw_name in candidates:
            canonical = re.split(r"\b(?:for|to|and|but|if|when|while|because|that|who|which)\b", raw_name, maxsplit=1)[0]
            canonical = canonical.strip(" ,.;:!?()[]{}\"'")
            object_id = _normalize_entity_key(canonical)

            if not object_id or object_id in _GENERIC_BLACKLIST:
                continue
            if len(object_id) < 3:
                continue

            rec = found.setdefault(
                object_id,
                {
                    "object_id": object_id,
                    "canonical_name": canonical,
                    "surface_forms": [],
                    "category": "object",
                    "physical_visualizable": True,
                    "direct_visual_evidence": False,
                    "mention_strength": 0.35,
                    "confidence": 0.30,
                    "evidence": [],
                    "source_channel": "dialogue",
                    "tracking_keep": False,
                },
            )
            if canonical not in rec["surface_forms"]:
                rec["surface_forms"].append(canonical)
            if sent not in rec["evidence"] and len(rec["evidence"]) < 3:
                rec["evidence"].append(sent)

    return list(found.values())


def _dialogue_inventory_records(
    scene,
    *,
    llm_model: str,
    cache_root: str,
    keep_categories: Sequence[str],
    dialogue_mode: str,
) -> List[Dict[str, Any]]:
    channel_texts = _scene_channel_texts(scene)
    dialogue_text = channel_texts["dialogue"]
    if not dialogue_text:
        return []

    if dialogue_mode == "none":
        return []

    if dialogue_mode in {"llm", "hybrid"}:
        try:
            entities = openai_scene_inventory_extract_entities(
                model=llm_model,
                scene_idx_non_omitted=scene.scene_idx_non_omitted,
                heading=scene.heading,
                channel_name="dialogue",
                channel_text=dialogue_text,
                cache_root=cache_root,
            )
            out: List[Dict[str, Any]] = []
            for ent in entities:
                if not ent.get("physical_visualizable", False):
                    continue

                category = str(ent.get("category", "other"))
                if category not in keep_categories:
                    continue

                canonical_name = str(ent.get("canonical_name", "")).strip()
                object_id = _normalize_entity_key(canonical_name)
                if not object_id:
                    continue

                out.append(
                    {
                        "object_id": object_id,
                        "canonical_name": canonical_name,
                        "surface_forms": list(ent.get("surface_forms", [])),
                        "category": category,
                        "physical_visualizable": True,
                        "direct_visual_evidence": bool(ent.get("direct_visual_evidence", False)),
                        "mention_strength": round(float(ent.get("mention_strength", 0.0)), 4),
                        "confidence": round(float(ent.get("confidence", 0.0)), 4),
                        "evidence": list(ent.get("evidence", []))[:3],
                        "source_channel": "dialogue",
                        "tracking_keep": False,
                    }
                )
            return out
        except Exception:
            if dialogue_mode == "llm":
                raise

    return _heuristic_dialogue_entities(dialogue_text)


def _merge_scene_inventory(
    action_objects: Sequence[Dict[str, Any]],
    dialogue_objects: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}

    for rec in list(action_objects) + list(dialogue_objects):
        aliases = set(_record_aliases(rec))
        matching_keys = [
            key
            for key, cur in merged.items()
            if aliases & cur["normalized_aliases"]
        ]
        merge_key = matching_keys[0] if matching_keys else None

        if merge_key is None:
            merge_key = rec["object_id"]
            merged[merge_key] = {
                "object_id": merge_key,
                "canonical_name": rec["canonical_name"],
                "surface_forms": [],
                "source_channels": [],
                "normalized_aliases": set(),
                "category_votes": Counter(),
                "physical_visualizable": False,
                "direct_visual_evidence": False,
                "mention_strength_max": 0.0,
                "confidence_max": 0.0,
                "evidence": [],
                "tracking_keep_any": False,
            }

        cur = merged[merge_key]
        for extra_key in matching_keys[1:]:
            extra = merged.pop(extra_key)
            if extra["canonical_name"] and len(extra["canonical_name"]) > len(cur["canonical_name"]):
                cur["canonical_name"] = extra["canonical_name"]
            for sf in extra["surface_forms"]:
                if sf not in cur["surface_forms"]:
                    cur["surface_forms"].append(sf)
            for channel in extra["source_channels"]:
                if channel not in cur["source_channels"]:
                    cur["source_channels"].append(channel)
            cur["normalized_aliases"].update(extra["normalized_aliases"])
            cur["category_votes"].update(extra["category_votes"])
            cur["physical_visualizable"] = cur["physical_visualizable"] or extra["physical_visualizable"]
            cur["direct_visual_evidence"] = cur["direct_visual_evidence"] or extra["direct_visual_evidence"]
            cur["mention_strength_max"] = max(cur["mention_strength_max"], extra["mention_strength_max"])
            cur["confidence_max"] = max(cur["confidence_max"], extra["confidence_max"])
            cur["tracking_keep_any"] = cur["tracking_keep_any"] or extra["tracking_keep_any"]
            for ev in extra["evidence"]:
                if ev not in cur["evidence"] and len(cur["evidence"]) < 5:
                    cur["evidence"].append(ev)

        if rec["canonical_name"] and len(rec["canonical_name"]) > len(cur["canonical_name"]):
            cur["canonical_name"] = rec["canonical_name"]
        for sf in rec.get("surface_forms", []):
            if sf not in cur["surface_forms"]:
                cur["surface_forms"].append(sf)
        if rec["source_channel"] not in cur["source_channels"]:
            cur["source_channels"].append(rec["source_channel"])
        cur["normalized_aliases"].update(aliases)
        cur["category_votes"][rec["category"]] += 1
        cur["physical_visualizable"] = cur["physical_visualizable"] or bool(rec["physical_visualizable"])
        cur["direct_visual_evidence"] = cur["direct_visual_evidence"] or bool(rec["direct_visual_evidence"])
        cur["mention_strength_max"] = max(cur["mention_strength_max"], float(rec["mention_strength"]))
        cur["confidence_max"] = max(cur["confidence_max"], float(rec["confidence"]))
        cur["tracking_keep_any"] = cur["tracking_keep_any"] or bool(rec.get("tracking_keep", False))
        for ev in rec.get("evidence", []):
            if ev not in cur["evidence"] and len(cur["evidence"]) < 5:
                cur["evidence"].append(ev)

    merged_list: List[Dict[str, Any]] = []
    for rec in merged.values():
        category = rec["category_votes"].most_common(1)[0][0] if rec["category_votes"] else "object"
        object_id = _normalize_entity_key(rec["canonical_name"]) or rec["object_id"]
        merged_list.append(
            {
                "object_id": object_id,
                "canonical_name": rec["canonical_name"],
                "surface_forms": rec["surface_forms"],
                "source_channels": sorted(rec["source_channels"]),
                "normalized_aliases": sorted(rec["normalized_aliases"]),
                "category": category,
                "physical_visualizable": rec["physical_visualizable"],
                "direct_visual_evidence": rec["direct_visual_evidence"],
                "mention_strength_max": round(rec["mention_strength_max"], 4),
                "confidence_max": round(rec["confidence_max"], 4),
                "evidence": rec["evidence"],
                "tracking_keep_any": rec["tracking_keep_any"],
            }
        )

    merged_list.sort(
        key=lambda rec: (
            rec["mention_strength_max"],
            rec["confidence_max"],
            rec["canonical_name"],
        ),
        reverse=True,
    )
    return merged_list


def build_scene_inventory(
    *,
    pdf: str,
    llm_model: str = "gpt-4o-mini",
    cache_dir: str = "data/llm_cache",
    keep_categories: Sequence[str] | None = None,
    dialogue_mode: str = "hybrid",
) -> Dict[str, Any]:
    """
    Build a high-recall double-channel scene inventory from the screenplay.

    Args:
        pdf: Path to the screenplay PDF.
        llm_model: OpenAI model name for extractors.
        cache_dir: Root cache directory.
        keep_categories: Allowed physical categories.
        dialogue_mode: One of {"llm", "heuristic", "hybrid", "none"}.

    Returns:
        A JSON-serializable scene inventory object.
    """
    if dialogue_mode not in {"llm", "heuristic", "hybrid", "none"}:
        raise ValueError(f"Unsupported dialogue_mode: {dialogue_mode}")

    keep_categories = keep_categories or [
        "object",
        "artifact",
        "weapon",
        "creature",
        "vehicle",
        "substance",
    ]

    scenes, meta = parse_pdf_scenes(pdf)
    scenes = [scene for scene in scenes if scene.scene_idx_non_omitted >= 0]

    out_scenes: Dict[str, Any] = {}
    for scene in scenes:
        action_objects = _action_inventory_records(
            scene,
            llm_model=llm_model,
            cache_root=cache_dir,
            keep_categories=keep_categories,
        )
        dialogue_objects = _dialogue_inventory_records(
            scene,
            llm_model=llm_model,
            cache_root=cache_dir,
            keep_categories=keep_categories,
            dialogue_mode=dialogue_mode,
        )
        merged_loose_objects = _merge_scene_inventory(action_objects, dialogue_objects)

        out_scenes[str(scene.scene_idx_non_omitted)] = {
            "heading": scene.heading,
            "page_start": scene.page_start,
            "page_end": scene.page_end,
            "action_objects": action_objects,
            "dialogue_objects": dialogue_objects,
            "merged_loose_objects": merged_loose_objects,
        }

    return {
        "meta": {
            "pdf": pdf,
            "scenes_non_omitted": len(scenes),
            "dialogue_mode": dialogue_mode,
            "llm_model": llm_model,
            "inventory_type": "high_recall_double_channel_scene_inventory",
        },
        "scenes": out_scenes,
    }
