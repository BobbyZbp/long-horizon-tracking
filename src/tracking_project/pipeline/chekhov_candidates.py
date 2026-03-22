"""
chekhov_candidates.py

This module implements the narrative layer that should exist before any
script-to-video alignment work begins.

Problem setting
---------------
Stage 1 is already a filter:
    - it focuses on long-gap re-occurrence events
    - it ranks precision-oriented object candidates
    - it can miss objects that matter for later Chekhov-style validation

To avoid filtering on top of an already filtered source, this module treats the
high-recall scene inventory as the primary candidate source, then uses Stage-1
events as a strong prior rather than a hard gate.

This module combines:
    - high-recall double-channel scene inventory (`scene_inventory.json`)
    - scene-level climax summaries (`scene_climax.json`)
    - optional Stage-1 re-occurrence events (`run1.json`)

and exports a smaller set of narrative candidates that are safer to use as the
upstream input for later alignment, frame sampling, and tracking demos.

Design notes
------------
- Deterministic: no new LLM calls are made here.
- High recall first: candidates are aggregated from scene inventory, not only
  from Stage-1 event rows.
- Stage-1 is supportive, not mandatory: unmatched inventory candidates can
  still survive if they exhibit a plausible setup -> absence -> payoff pattern.
- Explainable: each candidate includes validation reasons, setup/payoff scenes,
  inventory support, and the scene chain used to support the decision.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Sequence, Set, Tuple

from tracking_project.scoring.ranking import quantile


_GENERIC_ENTITY_ALIASES = {
    "animal",
    "artifact",
    "bird",
    "book",
    "box",
    "cabinet",
    "car",
    "creature",
    "figure",
    "item",
    "machine",
    "object",
    "person",
    "prop",
    "thing",
    "tool",
    "vehicle",
}


def _normalize_entity_key(name: str) -> str:
    text = name.strip().lower()
    text = re.sub(r"^(the|a|an)\s+", "", text)
    text = re.sub(r"[^\w\s']+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _is_generic_alias(name: str) -> bool:
    norm = _normalize_entity_key(name)
    if not norm:
        return True
    return norm in _GENERIC_ENTITY_ALIASES


def _specific_aliases(values: Sequence[str] | Set[str]) -> Set[str]:
    return {
        _normalize_entity_key(value)
        for value in values
        if _normalize_entity_key(value) and not _is_generic_alias(value)
    }


def _specific_overlap(left: Sequence[str] | Set[str], right: Sequence[str] | Set[str]) -> Set[str]:
    return _specific_aliases(set(left) & set(right))


def _alias_sets_match(left: Sequence[str] | Set[str], right: Sequence[str] | Set[str]) -> bool:
    """Return True only when two alias sets share a non-generic identifier.

    Generic head nouns like "book" or "car" are useful as weak evidence, but
    they are too broad to justify merging two inventory groups or attaching
    broad scene support. Without this guard, unrelated artifacts such as
    "Moste Potente Potions" and "diary" collapse into the same candidate.
    """
    return bool(_specific_overlap(left, right))


def _name_rank(name: str) -> Tuple[int, int, int]:
    norm = _normalize_entity_key(name)
    if not norm:
        return (-1, -1, -1)
    tokens = norm.split()
    return (
        0 if _is_generic_alias(norm) else 1,
        len(tokens),
        len(norm),
    )


def _prefer_name(candidate_name: str, current_name: str) -> bool:
    return _name_rank(candidate_name) > _name_rank(current_name)


def _event_aliases(event: Dict[str, Any]) -> Set[str]:
    aliases: Set[str] = set()
    for raw in [
        event.get("entity_key", ""),
        event.get("canonical_name", ""),
        *event.get("surface_forms", []),
    ]:
        norm = _normalize_entity_key(str(raw))
        if norm:
            aliases.add(norm)
    return aliases


def _inventory_aliases(obj_rec: Dict[str, Any]) -> Set[str]:
    aliases: Set[str] = set()
    for raw in [
        obj_rec.get("object_id", ""),
        obj_rec.get("canonical_name", ""),
        *obj_rec.get("surface_forms", []),
        *obj_rec.get("normalized_aliases", []),
    ]:
        norm = _normalize_entity_key(str(raw))
        if norm:
            aliases.add(norm)
    return aliases


def _gap_quantiles(events_json: Dict[str, Any]) -> Tuple[float, float]:
    meta = events_json.get("meta", {})
    gap_meta = meta.get("gap_quantiles", {})
    if "q50" in gap_meta and "q75" in gap_meta:
        return float(gap_meta["q50"]), float(gap_meta["q75"])

    gaps = [int(ev.get("gap_jump", 0)) for ev in events_json.get("events", [])]
    if not gaps:
        return 0.0, 0.0
    return quantile(gaps, 0.50), quantile(gaps, 0.75)


def _scene_map(scene_json: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    mapping: Dict[int, Dict[str, Any]] = {}
    for scene_idx, rec in scene_json.get("scenes", {}).items():
        mapping[int(scene_idx)] = rec
    return mapping


def _new_group(obj_rec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "canonical_name": str(obj_rec.get("canonical_name", "")).strip(),
        "entity_type": str(obj_rec.get("category", "object")),
        "alias_keys": set(),
        "category_votes": Counter(),
        "source_channels_seen": set(),
        "direct_visual_evidence_any": False,
        "mention_strength_values": [],
        "confidence_values": [],
        "scene_support_by_idx": {},
    }


def _add_support_record(
    support: Dict[str, Any],
    *,
    scene_idx: int,
    heading: str,
    obj_rec: Dict[str, Any],
) -> None:
    support["scene_idx_non_omitted"] = scene_idx
    support["heading"] = heading
    canonical_name = str(obj_rec.get("canonical_name", "")).strip()
    if canonical_name and canonical_name not in support["matched_inventory_objects"]:
        support["matched_inventory_objects"].append(canonical_name)

    for channel in obj_rec.get("source_channels", []) or [obj_rec.get("source_channel", "action")]:
        if channel and channel not in support["matched_source_channels"]:
            support["matched_source_channels"].append(channel)

    support["direct_visual_evidence"] = (
        support["direct_visual_evidence"] or bool(obj_rec.get("direct_visual_evidence", False))
    )
    support["mention_strength_max"] = max(
        support["mention_strength_max"],
        float(obj_rec.get("mention_strength_max", obj_rec.get("mention_strength", 0.0))),
    )
    support["confidence_max"] = max(
        support["confidence_max"],
        float(obj_rec.get("confidence_max", obj_rec.get("confidence", 0.0))),
    )

    for ev in obj_rec.get("evidence", []):
        if ev and ev not in support["inventory_evidence"] and len(support["inventory_evidence"]) < 4:
            support["inventory_evidence"].append(ev)


def _add_group_occurrence(
    group: Dict[str, Any],
    *,
    scene_idx: int,
    heading: str,
    obj_rec: Dict[str, Any],
) -> None:
    aliases = _inventory_aliases(obj_rec)
    group["alias_keys"].update(aliases)

    canonical_name = str(obj_rec.get("canonical_name", "")).strip()
    if canonical_name and _prefer_name(canonical_name, group["canonical_name"]):
        group["canonical_name"] = canonical_name

    entity_type = str(obj_rec.get("category", "object"))
    group["category_votes"][entity_type] += 1
    if group["category_votes"]:
        group["entity_type"] = group["category_votes"].most_common(1)[0][0]

    channels = obj_rec.get("source_channels", []) or [obj_rec.get("source_channel", "action")]
    group["source_channels_seen"].update(channels)
    group["direct_visual_evidence_any"] = (
        group["direct_visual_evidence_any"] or bool(obj_rec.get("direct_visual_evidence", False))
    )
    group["mention_strength_values"].append(
        float(obj_rec.get("mention_strength_max", obj_rec.get("mention_strength", 0.0)))
    )
    group["confidence_values"].append(
        float(obj_rec.get("confidence_max", obj_rec.get("confidence", 0.0)))
    )

    support = group["scene_support_by_idx"].setdefault(
        scene_idx,
        {
            "scene_idx_non_omitted": scene_idx,
            "heading": heading,
            "matched_inventory_objects": [],
            "matched_source_channels": [],
            "inventory_evidence": [],
            "direct_visual_evidence": False,
            "mention_strength_max": 0.0,
            "confidence_max": 0.0,
        },
    )
    _add_support_record(support, scene_idx=scene_idx, heading=heading, obj_rec=obj_rec)


def _merge_groups(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    if src["canonical_name"] and _prefer_name(src["canonical_name"], dst["canonical_name"]):
        dst["canonical_name"] = src["canonical_name"]

    dst["alias_keys"].update(src["alias_keys"])
    dst["category_votes"].update(src["category_votes"])
    if dst["category_votes"]:
        dst["entity_type"] = dst["category_votes"].most_common(1)[0][0]
    dst["source_channels_seen"].update(src["source_channels_seen"])
    dst["direct_visual_evidence_any"] = (
        dst["direct_visual_evidence_any"] or src["direct_visual_evidence_any"]
    )
    dst["mention_strength_values"].extend(src["mention_strength_values"])
    dst["confidence_values"].extend(src["confidence_values"])

    for scene_idx, support in src["scene_support_by_idx"].items():
        dst_support = dst["scene_support_by_idx"].setdefault(
            scene_idx,
            {
                "scene_idx_non_omitted": scene_idx,
                "heading": support.get("heading", ""),
                "matched_inventory_objects": [],
                "matched_source_channels": [],
                "inventory_evidence": [],
                "direct_visual_evidence": False,
                "mention_strength_max": 0.0,
                "confidence_max": 0.0,
            },
        )
        for name in support.get("matched_inventory_objects", []):
            if name not in dst_support["matched_inventory_objects"]:
                dst_support["matched_inventory_objects"].append(name)
        for channel in support.get("matched_source_channels", []):
            if channel not in dst_support["matched_source_channels"]:
                dst_support["matched_source_channels"].append(channel)
        for ev in support.get("inventory_evidence", []):
            if ev not in dst_support["inventory_evidence"] and len(dst_support["inventory_evidence"]) < 4:
                dst_support["inventory_evidence"].append(ev)
        dst_support["direct_visual_evidence"] = (
            dst_support["direct_visual_evidence"] or bool(support.get("direct_visual_evidence", False))
        )
        dst_support["mention_strength_max"] = max(
            dst_support["mention_strength_max"],
            float(support.get("mention_strength_max", 0.0)),
        )
        dst_support["confidence_max"] = max(
            dst_support["confidence_max"],
            float(support.get("confidence_max", 0.0)),
        )


def _collect_inventory_groups(scene_inventory_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    scene_by_idx = _scene_map(scene_inventory_json)
    groups: List[Dict[str, Any]] = []

    for scene_idx in sorted(scene_by_idx):
        scene_rec = scene_by_idx[scene_idx]
        heading = str(scene_rec.get("heading", ""))
        for obj_rec in scene_rec.get("merged_loose_objects", []):
            aliases = _inventory_aliases(obj_rec)
            if not aliases:
                continue

            matching_groups = [
                group
                for group in groups
                if _alias_sets_match(aliases, group["alias_keys"])
            ]
            if matching_groups:
                group = matching_groups[0]
                for extra_group in matching_groups[1:]:
                    _merge_groups(group, extra_group)
                    groups.remove(extra_group)
            else:
                group = _new_group(obj_rec)
                groups.append(group)

            _add_group_occurrence(group, scene_idx=scene_idx, heading=heading, obj_rec=obj_rec)

    return groups


def _event_anchor_scene_indices(event: Dict[str, Any] | None) -> Set[int]:
    if not event:
        return set()

    scene_indices = {
        int(idx)
        for idx in event.get("provenance", {}).get("scenes_non_omitted_list", [])
        if int(idx) >= 0
    }
    for field in ["scene_a_idx_non_omitted", "scene_b_idx_non_omitted"]:
        idx = int(event.get(field, -1))
        if idx >= 0:
            scene_indices.add(idx)
    return scene_indices


def _scan_inventory_support(
    aliases: Set[str],
    inventory_scene_by_idx: Dict[int, Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    scene_support_by_idx: Dict[int, Dict[str, Any]] = {}

    for scene_idx in sorted(inventory_scene_by_idx):
        scene_rec = inventory_scene_by_idx[scene_idx]
        heading = str(scene_rec.get("heading", ""))
        support = {
            "scene_idx_non_omitted": scene_idx,
            "heading": heading,
            "matched_inventory_objects": [],
            "matched_source_channels": [],
            "inventory_evidence": [],
            "direct_visual_evidence": False,
            "mention_strength_max": 0.0,
            "confidence_max": 0.0,
        }

        for obj_rec in scene_rec.get("merged_loose_objects", []):
            if _alias_sets_match(aliases, _inventory_aliases(obj_rec)):
                _add_support_record(support, scene_idx=scene_idx, heading=heading, obj_rec=obj_rec)

        if support["matched_inventory_objects"]:
            scene_support_by_idx[scene_idx] = support

    return scene_support_by_idx


def _best_stage1_event(
    aliases: Set[str],
    events: Sequence[Dict[str, Any]],
) -> Tuple[int | None, Dict[str, Any] | None, List[str]]:
    best_index = None
    best_event = None
    best_overlap: List[str] = []
    best_key = ((-1, -1, -1), -1, -1, -1, -1.0, -1.0)

    for event_index, event in enumerate(events):
        event_aliases = _event_aliases(event)
        overlap = sorted(aliases & event_aliases)
        if not overlap:
            continue

        specific_overlap = sorted(_specific_aliases(overlap))
        event_canonical = _normalize_entity_key(str(event.get("canonical_name", "")))
        event_key = _normalize_entity_key(str(event.get("entity_key", "")))
        canonical_hit = bool(event_canonical and event_canonical in aliases and not _is_generic_alias(event_canonical))
        key_hit = bool(event_key and event_key in aliases and not _is_generic_alias(event_key))

        # Generic head-noun overlap alone is too weak and causes object hijacking
        # such as "book" incorrectly attaching one artifact to another.
        if not (specific_overlap or canonical_hit or key_hit):
            continue

        key = (
            _name_rank(str(event.get("canonical_name", ""))),
            1 if canonical_hit else 0,
            1 if key_hit else 0,
            len(specific_overlap),
            float(event.get("gap_jump", 0)),
            float(event.get("scores", {}).get("confidence", 0.0)),
        )
        if key > best_key:
            best_key = key
            best_index = event_index
            best_event = event
            best_overlap = specific_overlap if specific_overlap else overlap

    return best_index, best_event, best_overlap


def _total_scenes(
    scene_inventory_json: Dict[str, Any],
    scene_climax_json: Dict[str, Any],
    inventory_scene_by_idx: Dict[int, Dict[str, Any]],
    climax_scene_by_idx: Dict[int, Dict[str, Any]],
) -> int:
    inventory_total = int(
        scene_inventory_json.get("meta", {}).get(
            "scenes_non_omitted",
            max(inventory_scene_by_idx) + 1 if inventory_scene_by_idx else 0,
        )
    )
    climax_total = int(
        scene_climax_json.get("meta", {}).get(
            "scenes_non_omitted",
            max(climax_scene_by_idx) + 1 if climax_scene_by_idx else 0,
        )
    )
    return max(inventory_total, climax_total, 1)


def _build_scene_chain(
    scene_support_by_idx: Dict[int, Dict[str, Any]],
    anchor_scene_indices: Set[int],
    inventory_scene_by_idx: Dict[int, Dict[str, Any]],
    climax_scene_by_idx: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    chain: List[Dict[str, Any]] = []
    scene_indices = sorted(set(scene_support_by_idx) | set(anchor_scene_indices))

    for scene_idx in scene_indices:
        inventory_scene = inventory_scene_by_idx.get(scene_idx, {})
        climax_scene = climax_scene_by_idx.get(scene_idx, {})
        support = scene_support_by_idx.get(scene_idx, {})

        match_source: List[str] = []
        if scene_idx in anchor_scene_indices:
            match_source.append("stage1_provenance")
        if support.get("matched_inventory_objects"):
            match_source.append("scene_inventory_match")

        heading = (
            support.get("heading")
            or inventory_scene.get("heading")
            or climax_scene.get("heading")
            or ""
        )
        chain.append(
            {
                "scene_idx_non_omitted": scene_idx,
                "heading": heading,
                "climax_score": int(climax_scene.get("climax_score", 0)),
                "matched_inventory_objects": list(support.get("matched_inventory_objects", [])),
                "matched_source_channels": sorted(support.get("matched_source_channels", [])),
                "inventory_evidence": list(support.get("inventory_evidence", [])),
                "direct_visual_evidence": bool(support.get("direct_visual_evidence", False)),
                "mention_strength_max": round(float(support.get("mention_strength_max", 0.0)), 4),
                "confidence_max": round(float(support.get("confidence_max", 0.0)), 4),
                "match_source": match_source,
            }
        )

    return chain


def _scene_gap_stats(scene_indices: Sequence[int]) -> Dict[str, Any]:
    unique = sorted(set(int(idx) for idx in scene_indices))
    if len(unique) < 2:
        raise ValueError("Need at least two scene indices to compute narrative gap statistics")

    best_gap = -1
    scene_a = unique[0]
    scene_b = unique[-1]
    for left, right in zip(unique, unique[1:]):
        gap = right - left
        if gap > best_gap:
            best_gap = gap
            scene_a = left
            scene_b = right

    span = unique[-1] - unique[0] + 1
    coverage = len(unique) / float(max(span, 1))
    return {
        "scene_a_idx_non_omitted": scene_a,
        "scene_b_idx_non_omitted": scene_b,
        "gap_jump": best_gap,
        "gap_absent": max(best_gap - 1, 0),
        "freq_scenes": len(unique),
        "coverage": coverage,
    }


def _candidate_stats_from_stage1_event(event: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "scene_a_idx_non_omitted": int(event.get("scene_a_idx_non_omitted", -1)),
        "scene_b_idx_non_omitted": int(event.get("scene_b_idx_non_omitted", -1)),
        "gap_jump": int(event.get("gap_jump", 0)),
        "gap_absent": int(event.get("gap_absent", 0)),
        "freq_scenes": int(event.get("freq_scenes", 0)),
        "coverage": float(event.get("coverage", 0.0)),
        "scores": {
            "narrative_action_centrality": float(
                event.get("scores", {}).get("narrative_action_centrality", 0.0)
            ),
            "visual_grounded_score": float(
                event.get("scores", {}).get("visual_grounded_score", 0.0)
            ),
            "state_change_potential": float(
                event.get("scores", {}).get("state_change_potential", 0.0)
            ),
            "confidence": float(event.get("scores", {}).get("confidence", 0.0)),
        },
    }


def _candidate_stats_from_inventory(
    *,
    entity_type: str,
    scene_chain: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    scene_indices = [rec["scene_idx_non_omitted"] for rec in scene_chain]
    gap_stats = _scene_gap_stats(scene_indices)

    strengths = [float(rec.get("mention_strength_max", 0.0)) for rec in scene_chain]
    confidences = [float(rec.get("confidence_max", 0.0)) for rec in scene_chain]
    action_strengths = [
        float(rec.get("mention_strength_max", 0.0))
        for rec in scene_chain
        if "action" in rec.get("matched_source_channels", [])
    ]
    visual_votes = [
        1.0
        if rec.get("direct_visual_evidence", False)
        else 0.4
        if rec.get("matched_source_channels") == ["dialogue"]
        else 0.65
        if "action" in rec.get("matched_source_channels", [])
        else 0.0
        for rec in scene_chain
    ]

    if entity_type in {"artifact", "weapon", "substance"}:
        state_score = 0.45
    elif entity_type in {"object", "vehicle"}:
        state_score = 0.35
    elif entity_type == "creature":
        state_score = 0.25
    else:
        state_score = 0.20

    action_score = max(action_strengths) if action_strengths else (max(strengths) * 0.65 if strengths else 0.0)
    visual_score = sum(visual_votes) / float(max(len(visual_votes), 1))
    confidence_score = sum(confidences) / float(max(len(confidences), 1))

    return {
        **gap_stats,
        "scores": {
            "narrative_action_centrality": round(_clamp(action_score, 0.0, 1.0), 4),
            "visual_grounded_score": round(_clamp(visual_score, 0.0, 1.0), 4),
            "state_change_potential": round(_clamp(state_score, 0.0, 1.0), 4),
            "confidence": round(_clamp(confidence_score, 0.0, 1.0), 4),
        },
    }


def _validation_reasons(
    *,
    candidate_source: str,
    total_scenes: int,
    first_scene_idx: int,
    last_scene_idx: int,
    gap_jump: int,
    q50: float,
    q75: float,
    max_payoff_climax: int,
    coverage: float,
    action: float,
    state: float,
    has_stage1_prior: bool,
    has_loose_reinforcement: bool,
    has_action_reinforcement: bool,
    has_dialogue_only_support: bool,
) -> List[str]:
    reasons: List[str] = []

    if candidate_source == "inventory_only":
        reasons.append("inventory_only_candidate")
    elif candidate_source == "stage1_fallback":
        reasons.append("stage1_fallback_candidate")

    if has_stage1_prior:
        reasons.append("stage1_prior_support")
    if total_scenes > 1 and first_scene_idx <= 0.35 * (total_scenes - 1):
        reasons.append("early_setup_scene")
    if q75 > 0 and gap_jump >= q75:
        reasons.append("extreme_reappearance_gap")
    elif q50 > 0 and gap_jump >= q50:
        reasons.append("long_reappearance_gap")
    if total_scenes > 1 and last_scene_idx >= 0.65 * (total_scenes - 1):
        reasons.append("late_story_return")
    if max_payoff_climax >= 70:
        reasons.append("high_climax_payoff_scene")
    if coverage <= 0.35:
        reasons.append("sparse_reoccurrence_pattern")
    if action >= 0.55:
        reasons.append("action_central_object")
    if state >= 0.35:
        reasons.append("state_changeable_object")
    if has_loose_reinforcement:
        reasons.append("scene_inventory_loose_reinforcement")
    if has_action_reinforcement:
        reasons.append("action_channel_reinforcement")
    if has_dialogue_only_support:
        reasons.append("dialogue_only_support")

    return reasons


def _validation_label(score: float) -> str:
    if score >= 0.72:
        return "high"
    if score >= 0.58:
        return "medium"
    if score >= 0.45:
        return "low"
    return "unlikely"


def _score_candidate(
    *,
    candidate_source: str,
    candidate_stats: Dict[str, Any],
    scene_chain: List[Dict[str, Any]],
    total_scenes: int,
    q50: float,
    q75: float,
    has_stage1_prior: bool,
) -> Dict[str, Any]:
    scene_indices = [rec["scene_idx_non_omitted"] for rec in scene_chain]
    first_scene_idx = min(scene_indices)
    last_scene_idx = max(scene_indices)

    scene_a = int(candidate_stats.get("scene_a_idx_non_omitted", first_scene_idx))
    scene_b = int(candidate_stats.get("scene_b_idx_non_omitted", last_scene_idx))
    gap_jump = int(candidate_stats.get("gap_jump", 0))
    coverage = float(candidate_stats.get("coverage", 1.0))

    setup_scene_indices = sorted(idx for idx in scene_indices if idx <= scene_a)[:2]
    if not setup_scene_indices:
        setup_scene_indices = [first_scene_idx]

    payoff_scene_indices = sorted(idx for idx in scene_indices if idx >= scene_b)
    if not payoff_scene_indices:
        payoff_scene_indices = [last_scene_idx]

    scores = candidate_stats.get("scores", {})
    action = float(scores.get("narrative_action_centrality", 0.0))
    visual = float(scores.get("visual_grounded_score", 0.0))
    state = float(scores.get("state_change_potential", 0.0))
    confidence = float(scores.get("confidence", 0.0))
    freq_scenes = int(candidate_stats.get("freq_scenes", len(scene_indices)))

    payoff_climax_values = [
        int(rec.get("climax_score", 0))
        for rec in scene_chain
        if rec["scene_idx_non_omitted"] >= scene_b
    ]
    max_payoff_climax = max(payoff_climax_values) if payoff_climax_values else int(scene_chain[-1].get("climax_score", 0))

    has_loose_reinforcement = any(
        ("scene_inventory_match" in rec["match_source"]) and ("stage1_provenance" not in rec["match_source"])
        for rec in scene_chain
    )
    has_action_reinforcement = any(
        ("scene_inventory_match" in rec["match_source"])
        and ("stage1_provenance" not in rec["match_source"])
        and ("action" in rec.get("matched_source_channels", []))
        for rec in scene_chain
    )
    has_dialogue_only_support = any(
        rec.get("matched_source_channels", []) == ["dialogue"]
        for rec in scene_chain
    )

    denom = float(max(total_scenes - 1, 1))
    setup_early = _clamp(1.0 - first_scene_idx / denom, 0.0, 1.0)
    payoff_late = _clamp(last_scene_idx / denom, 0.0, 1.0)
    gap_norm = _clamp(gap_jump / float(max(q75, 1.0)), 0.0, 1.0)
    payoff_climax_norm = _clamp(max_payoff_climax / 100.0, 0.0, 1.0)
    sparse_norm = _clamp(1.0 - coverage, 0.0, 1.0)
    recurrence_focus = 1.0 if freq_scenes <= 5 else _clamp(1.0 - (freq_scenes - 5) / 10.0, 0.0, 1.0)

    score = (
        0.22 * gap_norm
        + 0.15 * setup_early
        + 0.15 * payoff_late
        + 0.16 * payoff_climax_norm
        + 0.10 * action
        + 0.08 * visual
        + 0.05 * state
        + 0.04 * confidence
        + 0.04 * sparse_norm
        + 0.03 * recurrence_focus
        + (0.04 if has_loose_reinforcement else 0.0)
        + (0.02 if has_action_reinforcement else 0.0)
        + (0.03 if has_stage1_prior else 0.0)
    )
    score = _clamp(score, 0.0, 1.0)

    reasons = _validation_reasons(
        candidate_source=candidate_source,
        total_scenes=total_scenes,
        first_scene_idx=first_scene_idx,
        last_scene_idx=last_scene_idx,
        gap_jump=gap_jump,
        q50=q50,
        q75=q75,
        max_payoff_climax=max_payoff_climax,
        coverage=coverage,
        action=action,
        state=state,
        has_stage1_prior=has_stage1_prior,
        has_loose_reinforcement=has_loose_reinforcement,
        has_action_reinforcement=has_action_reinforcement,
        has_dialogue_only_support=has_dialogue_only_support,
    )

    role_by_scene: Dict[int, List[str]] = {idx: [] for idx in scene_indices}
    for idx in setup_scene_indices:
        role_by_scene[idx].append("setup_candidate")
    for idx in payoff_scene_indices:
        role_by_scene[idx].append("payoff_candidate")
    if scene_a in role_by_scene:
        role_by_scene[scene_a].append("gap_anchor_a")
    if scene_b in role_by_scene:
        role_by_scene[scene_b].append("gap_anchor_b")

    for rec in scene_chain:
        if rec["climax_score"] >= 70:
            role_by_scene[rec["scene_idx_non_omitted"]].append("high_climax_scene")
        if "scene_inventory_match" in rec["match_source"] and "stage1_provenance" not in rec["match_source"]:
            role_by_scene[rec["scene_idx_non_omitted"]].append("scene_inventory_loose_match")
        if "action" in rec.get("matched_source_channels", []):
            role_by_scene[rec["scene_idx_non_omitted"]].append("action_supported_match")
        elif rec.get("matched_source_channels", []) == ["dialogue"]:
            role_by_scene[rec["scene_idx_non_omitted"]].append("dialogue_only_match")

    chain_with_roles = []
    for rec in scene_chain:
        rec2 = dict(rec)
        rec2["role_tags"] = role_by_scene.get(rec["scene_idx_non_omitted"], [])
        chain_with_roles.append(rec2)

    return {
        "possible_chekhov_score": round(score, 4),
        "possible_chekhov_label": _validation_label(score),
        "validation_reasons": reasons,
        "setup_scene_indices": setup_scene_indices,
        "payoff_scene_indices": payoff_scene_indices,
        "loose_scene_chain": chain_with_roles,
    }


def _inventory_support_summary(scene_support_by_idx: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    source_channels_seen = sorted(
        {
            channel
            for support in scene_support_by_idx.values()
            for channel in support.get("matched_source_channels", [])
        }
    )
    return {
        "inventory_scene_indices": sorted(scene_support_by_idx),
        "inventory_scene_count": len(scene_support_by_idx),
        "source_channels_seen": source_channels_seen,
        "direct_visual_evidence_any": any(
            bool(support.get("direct_visual_evidence", False))
            for support in scene_support_by_idx.values()
        ),
    }


def _stage1_prior_trace(
    event_index: int | None,
    event: Dict[str, Any] | None,
    matched_aliases: Sequence[str],
) -> Dict[str, Any] | None:
    if event is None or event_index is None:
        return None

    return {
        "source_event_index": event_index,
        "matched_aliases": list(matched_aliases),
        "canonical_name": str(event.get("canonical_name", "")),
        "entity_type": str(event.get("entity_type", "object")),
        "scene_a_idx_non_omitted": int(event.get("scene_a_idx_non_omitted", -1)),
        "scene_b_idx_non_omitted": int(event.get("scene_b_idx_non_omitted", -1)),
        "gap_jump": int(event.get("gap_jump", 0)),
        "gap_absent": int(event.get("gap_absent", 0)),
        "freq_scenes": int(event.get("freq_scenes", 0)),
        "coverage": float(event.get("coverage", 0.0)),
        "scores": {
            "narrative_action_centrality": float(
                event.get("scores", {}).get("narrative_action_centrality", 0.0)
            ),
            "visual_grounded_score": float(
                event.get("scores", {}).get("visual_grounded_score", 0.0)
            ),
            "state_change_potential": float(
                event.get("scores", {}).get("state_change_potential", 0.0)
            ),
            "confidence": float(event.get("scores", {}).get("confidence", 0.0)),
        },
    }


def _candidate_object_id(canonical_name: str, aliases: Set[str]) -> str:
    canonical_id = _normalize_entity_key(canonical_name)
    specific_aliases = sorted(
        _specific_aliases(aliases),
        key=_name_rank,
        reverse=True,
    )
    if canonical_id and not _is_generic_alias(canonical_id):
        return canonical_id
    if specific_aliases:
        return specific_aliases[0]
    return canonical_id or (sorted(aliases)[0] if aliases else "")


def _choose_canonical_name(
    group: Dict[str, Any],
    matched_event: Dict[str, Any] | None,
) -> str:
    canonical_name = group["canonical_name"]
    event_name = str(matched_event.get("canonical_name", "")).strip() if matched_event else ""
    if event_name and _prefer_name(event_name, canonical_name):
        return event_name
    return canonical_name


def _merge_inventory_support_records(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    scene_indices = sorted(
        set(left.get("inventory_scene_indices", [])) | set(right.get("inventory_scene_indices", []))
    )
    source_channels = sorted(
        set(left.get("source_channels_seen", [])) | set(right.get("source_channels_seen", []))
    )
    return {
        "inventory_scene_indices": scene_indices,
        "inventory_scene_count": len(scene_indices),
        "source_channels_seen": source_channels,
        "direct_visual_evidence_any": bool(
            left.get("direct_visual_evidence_any", False)
            or right.get("direct_visual_evidence_any", False)
        ),
    }


def _merge_scene_chain(
    left: Sequence[Dict[str, Any]],
    right: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_scene: Dict[int, Dict[str, Any]] = {}
    for rec in list(left) + list(right):
        scene_idx = int(rec["scene_idx_non_omitted"])
        dst = by_scene.setdefault(
            scene_idx,
            {
                "scene_idx_non_omitted": scene_idx,
                "heading": rec.get("heading", ""),
                "climax_score": int(rec.get("climax_score", 0)),
                "matched_inventory_objects": [],
                "matched_source_channels": [],
                "inventory_evidence": [],
                "direct_visual_evidence": False,
                "mention_strength_max": 0.0,
                "confidence_max": 0.0,
                "match_source": [],
                "role_tags": [],
            },
        )
        if rec.get("heading") and not dst.get("heading"):
            dst["heading"] = rec["heading"]
        dst["climax_score"] = max(int(dst.get("climax_score", 0)), int(rec.get("climax_score", 0)))
        dst["direct_visual_evidence"] = bool(
            dst.get("direct_visual_evidence", False) or rec.get("direct_visual_evidence", False)
        )
        dst["mention_strength_max"] = max(
            float(dst.get("mention_strength_max", 0.0)),
            float(rec.get("mention_strength_max", 0.0)),
        )
        dst["confidence_max"] = max(
            float(dst.get("confidence_max", 0.0)),
            float(rec.get("confidence_max", 0.0)),
        )
        for field in [
            "matched_inventory_objects",
            "matched_source_channels",
            "inventory_evidence",
            "match_source",
            "role_tags",
        ]:
            for value in rec.get(field, []):
                if value not in dst[field]:
                    dst[field].append(value)

    return [by_scene[idx] for idx in sorted(by_scene)]


def _candidate_aliases(candidate: Dict[str, Any]) -> Set[str]:
    aliases: Set[str] = set()
    for raw in [candidate.get("object_id", ""), candidate.get("canonical_name", "")]:
        norm = _normalize_entity_key(str(raw))
        if norm:
            aliases.add(norm)

    prior = candidate.get("stage1_prior") or {}
    for raw in [prior.get("canonical_name", ""), *prior.get("matched_aliases", [])]:
        norm = _normalize_entity_key(str(raw))
        if norm:
            aliases.add(norm)
    for prior_rec in candidate.get("stage1_supports", []):
        for raw in [prior_rec.get("canonical_name", ""), *prior_rec.get("matched_aliases", [])]:
            norm = _normalize_entity_key(str(raw))
            if norm:
                aliases.add(norm)
    return aliases


def _candidate_match(candidate: Dict[str, Any], other: Dict[str, Any]) -> bool:
    if candidate.get("object_id") == other.get("object_id"):
        return True
    overlap = _candidate_aliases(candidate) & _candidate_aliases(other)
    return bool(_specific_aliases(overlap))


def _merged_candidate_source(values: Set[str]) -> str:
    if "inventory_with_stage1_prior" in values:
        return "inventory_with_stage1_prior"
    if "inventory_only" in values and "stage1_fallback" in values:
        return "inventory_with_stage1_prior"
    if "inventory_only" in values:
        return "inventory_only"
    return "stage1_fallback"


def _prior_rank(prior: Dict[str, Any] | None) -> Tuple[int, float, Tuple[int, int, int]]:
    if not prior:
        return (-1, -1.0, (-1, -1, -1))
    return (
        int(prior.get("gap_jump", 0)),
        float(prior.get("scores", {}).get("confidence", 0.0)),
        _name_rank(str(prior.get("canonical_name", ""))),
    )


def _merge_stage1_supports(
    left: Sequence[Dict[str, Any]],
    right: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged: Dict[int, Dict[str, Any]] = {}
    for rec in list(left) + list(right):
        event_index = int(rec.get("source_event_index", -1))
        if event_index < 0:
            continue
        cur = merged.get(event_index)
        if cur is None or _prior_rank(rec) > _prior_rank(cur):
            merged[event_index] = rec
    return [merged[idx] for idx in sorted(merged)]


def _merge_two_candidates(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    primary = left if (
        float(left.get("possible_chekhov_score", 0.0)),
        int(left.get("candidate_stats", {}).get("gap_jump", 0)),
    ) >= (
        float(right.get("possible_chekhov_score", 0.0)),
        int(right.get("candidate_stats", {}).get("gap_jump", 0)),
    ) else right
    secondary = right if primary is left else left

    stage1_supports = _merge_stage1_supports(
        left.get("stage1_supports", []),
        right.get("stage1_supports", []),
    )
    best_prior = None
    for prior in stage1_supports:
        if best_prior is None or _prior_rank(prior) > _prior_rank(best_prior):
            best_prior = prior

    merged_chain = _merge_scene_chain(
        left.get("loose_scene_chain", []),
        right.get("loose_scene_chain", []),
    )
    canonical_name = primary.get("canonical_name", "")
    if _prefer_name(secondary.get("canonical_name", ""), canonical_name):
        canonical_name = secondary.get("canonical_name", "")
    if best_prior and _prefer_name(str(best_prior.get("canonical_name", "")), canonical_name):
        canonical_name = str(best_prior.get("canonical_name", ""))

    merged = dict(primary)
    merged["canonical_name"] = canonical_name
    merged["object_id"] = _candidate_object_id(
        canonical_name,
        _candidate_aliases(left) | _candidate_aliases(right),
    )
    merged["candidate_source"] = _merged_candidate_source(
        {str(left.get("candidate_source", "")), str(right.get("candidate_source", ""))}
    )
    merged["possible_chekhov_score"] = round(
        max(float(left.get("possible_chekhov_score", 0.0)), float(right.get("possible_chekhov_score", 0.0))),
        4,
    )
    merged["possible_chekhov_label"] = _validation_label(float(merged["possible_chekhov_score"]))
    merged["validation_reasons"] = []
    for reason in list(left.get("validation_reasons", [])) + list(right.get("validation_reasons", [])):
        if reason not in merged["validation_reasons"]:
            merged["validation_reasons"].append(reason)
    if "dedup_merged_support" not in merged["validation_reasons"]:
        merged["validation_reasons"].append("dedup_merged_support")
    merged["setup_scene_indices"] = sorted(
        set(left.get("setup_scene_indices", [])) | set(right.get("setup_scene_indices", []))
    )
    merged["payoff_scene_indices"] = sorted(
        set(left.get("payoff_scene_indices", [])) | set(right.get("payoff_scene_indices", []))
    )
    merged["loose_scene_chain"] = merged_chain
    merged["inventory_support"] = _merge_inventory_support_records(
        left.get("inventory_support", {}),
        right.get("inventory_support", {}),
    )
    merged["stage1_supports"] = stage1_supports
    merged["stage1_prior"] = best_prior
    return merged


def _dedup_candidates(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    for candidate in candidates:
        merged = False
        for idx, existing in enumerate(deduped):
            if _candidate_match(existing, candidate):
                deduped[idx] = _merge_two_candidates(existing, candidate)
                merged = True
                break
        if not merged:
            deduped.append(candidate)
    return deduped


def build_chekhov_candidates(
    events_json: Dict[str, Any],
    scene_inventory_json: Dict[str, Any],
    scene_climax_json: Dict[str, Any],
    *,
    min_validation_score: float = 0.55,
    top_k: int | None = None,
) -> Dict[str, Any]:
    """
    Build loose narrative candidates and validate likely Chekhov patterns.

    Args:
        events_json: Stage-1 events object loaded from JSON. Used as a prior,
            not the only candidate source.
        scene_inventory_json: High-recall double-channel scene inventory object.
        scene_climax_json: Scene-level climax summaries.
        min_validation_score: Minimum possible-Chekhov score required to keep a
            candidate in the final output.
        top_k: Optional cap on returned candidates after sorting.

    Returns:
        A JSON-serializable dict with `meta`, `field_spec`, and `candidates`.
    """
    events = list(events_json.get("events", []))
    inventory_scene_by_idx = _scene_map(scene_inventory_json)
    climax_scene_by_idx = _scene_map(scene_climax_json)

    if not inventory_scene_by_idx:
        raise ValueError("scene_inventory_json contains no scene inventory")
    if not climax_scene_by_idx:
        raise ValueError("scene_climax_json contains no scene summaries")

    q50, q75 = _gap_quantiles(events_json)
    total_scenes = _total_scenes(
        scene_inventory_json,
        scene_climax_json,
        inventory_scene_by_idx,
        climax_scene_by_idx,
    )

    inventory_groups = _collect_inventory_groups(scene_inventory_json)
    matched_stage1_indices: Set[int] = set()
    candidates: List[Dict[str, Any]] = []

    for group in inventory_groups:
        event_index, matched_event, overlap_aliases = _best_stage1_event(group["alias_keys"], events)
        if event_index is not None:
            matched_stage1_indices.add(event_index)

        scene_chain = _build_scene_chain(
            group["scene_support_by_idx"],
            _event_anchor_scene_indices(matched_event),
            inventory_scene_by_idx,
            climax_scene_by_idx,
        )
        if len(scene_chain) < 2:
            continue

        candidate_source = "inventory_with_stage1_prior" if matched_event else "inventory_only"
        candidate_stats = (
            _candidate_stats_from_stage1_event(matched_event)
            if matched_event is not None
            else _candidate_stats_from_inventory(
                entity_type=group["entity_type"],
                scene_chain=scene_chain,
            )
        )
        validation = _score_candidate(
            candidate_source=candidate_source,
            candidate_stats=candidate_stats,
            scene_chain=scene_chain,
            total_scenes=total_scenes,
            q50=q50,
            q75=q75,
            has_stage1_prior=matched_event is not None,
        )
        if validation["possible_chekhov_score"] < min_validation_score:
            continue

        canonical_name = _choose_canonical_name(group, matched_event)
        stage1_prior = _stage1_prior_trace(event_index, matched_event, overlap_aliases)

        candidates.append(
            {
                "object_id": _candidate_object_id(canonical_name, group["alias_keys"]),
                "canonical_name": canonical_name,
                "entity_type": (
                    str(matched_event.get("entity_type", group["entity_type"]))
                    if matched_event
                    else group["entity_type"]
                ),
                "candidate_source": candidate_source,
                "candidate_stats": candidate_stats,
                "inventory_support": _inventory_support_summary(group["scene_support_by_idx"]),
                "stage1_prior": stage1_prior,
                "stage1_supports": [stage1_prior] if stage1_prior else [],
                **validation,
            }
        )

    for event_index, event in enumerate(events):
        if event_index in matched_stage1_indices:
            continue

        aliases = _event_aliases(event)
        scene_support_by_idx = _scan_inventory_support(aliases, inventory_scene_by_idx)
        scene_chain = _build_scene_chain(
            scene_support_by_idx,
            _event_anchor_scene_indices(event),
            inventory_scene_by_idx,
            climax_scene_by_idx,
        )
        if len(scene_chain) < 2:
            continue

        candidate_stats = _candidate_stats_from_stage1_event(event)
        validation = _score_candidate(
            candidate_source="stage1_fallback",
            candidate_stats=candidate_stats,
            scene_chain=scene_chain,
            total_scenes=total_scenes,
            q50=q50,
            q75=q75,
            has_stage1_prior=True,
        )
        if validation["possible_chekhov_score"] < min_validation_score:
            continue

        stage1_prior = _stage1_prior_trace(event_index, event, sorted(aliases))
        candidates.append(
            {
                "object_id": str(event.get("entity_key") or _normalize_entity_key(str(event.get("canonical_name", "")))),
                "canonical_name": str(event.get("canonical_name", "")),
                "entity_type": str(event.get("entity_type", "object")),
                "candidate_source": "stage1_fallback",
                "candidate_stats": candidate_stats,
                "inventory_support": _inventory_support_summary(scene_support_by_idx),
                "stage1_prior": stage1_prior,
                "stage1_supports": [stage1_prior] if stage1_prior else [],
                **validation,
            }
        )

    candidates = _dedup_candidates(candidates)
    candidates.sort(
        key=lambda rec: (
            rec["possible_chekhov_score"],
            rec["candidate_stats"]["gap_jump"],
            rec["candidate_stats"]["scores"]["narrative_action_centrality"],
            rec["candidate_stats"]["scores"]["visual_grounded_score"],
        ),
        reverse=True,
    )
    if top_k is not None:
        candidates = candidates[:top_k]

    source_counter = Counter(rec["candidate_source"] for rec in candidates)
    return {
        "meta": {
            "stage1_events_considered": len(events),
            "inventory_groups_considered": len(inventory_groups),
            "scene_inventory_summaries_considered": len(inventory_scene_by_idx),
            "scene_climax_summaries_considered": len(climax_scene_by_idx),
            "candidates_returned": len(candidates),
            "candidate_source_counts": dict(source_counter),
            "min_validation_score": min_validation_score,
            "gap_quantiles": {"q50": q50, "q75": q75},
            "validation_type": "scene_level_loose_candidate_mining_plus_cross_scene_narrative_validation",
        },
        "field_spec": _build_field_spec(),
        "candidates": candidates,
    }


def _build_field_spec() -> Dict[str, Any]:
    """
    Build a self-documenting field specification for narrative candidate output.
    """
    return {
        "object_id": {
            "type": "string",
            "description": "Normalized object identifier used as the stable candidate key.",
        },
        "canonical_name": {
            "type": "string",
            "description": "Human-readable canonical name chosen from inventory evidence and optional Stage-1 prior.",
        },
        "entity_type": {
            "type": "string",
            "description": "Dominant object category for the candidate.",
        },
        "candidate_source": {
            "type": "string",
            "description": (
                "Where the candidate came from: inventory_with_stage1_prior / "
                "inventory_only / stage1_fallback."
            ),
        },
        "candidate_stats": {
            "type": "object",
            "description": "Gap and semantic scores actually used for validation and ranking.",
        },
        "inventory_support": {
            "type": "object",
            "description": "Summary of scene-inventory support: scene count, source channels, and direct-visual-evidence flag.",
        },
        "stage1_prior": {
            "type": "object|null",
            "description": "Matched Stage-1 event used as a prior, or null if this is inventory-only.",
        },
        "stage1_supports": {
            "type": "array",
            "description": "All distinct Stage-1 prior traces merged into this candidate after deduplication.",
        },
        "possible_chekhov_score": {
            "type": "float",
            "description": (
                "Heuristic score in [0,1] estimating whether the object exhibits a "
                "setup -> absence -> payoff pattern across scenes."
            ),
        },
        "possible_chekhov_label": {
            "type": "string",
            "description": "Discrete bucket derived from possible_chekhov_score: high / medium / low / unlikely.",
        },
        "validation_reasons": {
            "type": "list[string]",
            "description": "Human-readable reason tags explaining why the object was kept as a narrative candidate.",
        },
        "setup_scene_indices": {
            "type": "list[int]",
            "description": "Earliest scene indices that serve as setup-side support for the candidate.",
        },
        "payoff_scene_indices": {
            "type": "list[int]",
            "description": "Later scene indices that serve as payoff-side support for the candidate.",
        },
        "loose_scene_chain": {
            "type": "list[object]",
            "description": (
                "Expanded scene chain for the candidate. This includes inventory matches, "
                "matched source channels, climax scores, and optional Stage-1 anchor scenes."
            ),
        },
    }
