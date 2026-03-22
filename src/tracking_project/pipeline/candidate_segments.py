"""
candidate_segments.py

This module implements the candidate segment filtering layer after coarse
script-to-video alignment and before frame sampling.

Why this file exists
--------------------
The coarse alignment layer intentionally remains broad. It maps narrative scene steps to
approximate video windows, but those windows are still too fine-grained and too
numerous to serve directly as tracking inputs.

Candidate segment filtering therefore exists as its own layer:
    - prior layer: coarse alignment
    - current layer: segment filtering / consolidation
    - next layer: frame sampling

This module:
    - groups aligned scene windows by object candidate
    - expands each aligned window into a search segment
    - merges nearby windows when they describe the same visual search region
    - assigns reasons and a priority score
    - caps the number of segments per object so downstream tracking stays
      focused and affordable

The output of this module is the canonical candidate-segment JSON.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _candidate_lookup(candidates_json: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {
        idx: candidate
        for idx, candidate in enumerate(candidates_json.get("candidates", []))
    }


def _scene_role_map(candidate: Dict[str, Any]) -> Dict[int, List[str]]:
    role_map: Dict[int, List[str]] = {}
    for scene_rec in candidate.get("loose_scene_chain", []):
        role_map[int(scene_rec["scene_idx_non_omitted"])] = list(scene_rec.get("role_tags", []))
    return role_map


def _initial_reason_tags(
    *,
    candidate_source: str,
    role_tags: Sequence[str],
    component_scores: Dict[str, Any],
) -> List[str]:
    reasons: List[str] = []
    if candidate_source:
        reasons.append(f"candidate_source:{candidate_source}")
    for role in role_tags:
        reasons.append(f"role:{role}")
    if float(component_scores.get("object_overlap", 0.0)) > 0.0:
        reasons.append("subtitle_object_overlap")
    if float(component_scores.get("lexical_overlap", 0.0)) > 0.0:
        reasons.append("subtitle_lexical_overlap")
    if not reasons:
        reasons.append("expected_time_fallback")
    return reasons


def _segment_role(role_tags: Sequence[str]) -> str:
    if "payoff_candidate" in role_tags:
        return "payoff"
    if "setup_candidate" in role_tags:
        return "setup"
    if "scene_inventory_loose_match" in role_tags:
        return "reappearance"
    return "bridge"


def _priority_score(
    *,
    candidate_score: float,
    alignment_confidence: float,
    role_tags: Sequence[str],
    candidate_source: str,
) -> float:
    role_bonus = 0.0
    if "payoff_candidate" in role_tags:
        role_bonus += 0.14
    if "setup_candidate" in role_tags:
        role_bonus += 0.08
    if "high_climax_scene" in role_tags:
        role_bonus += 0.10
    if "action_supported_match" in role_tags:
        role_bonus += 0.05
    if candidate_source == "inventory_only":
        role_bonus -= 0.02

    score = (
        0.56 * float(candidate_score)
        + 0.34 * float(alignment_confidence)
        + role_bonus
    )
    return round(_clamp(score, 0.0, 1.0), 4)


def _has_object_specific_alignment(alignment_rec: Dict[str, Any]) -> bool:
    """Require matched subtitle evidence to actually point at the object.

    After the alignment layer was tightened, a small set of windows could still
    survive using only generic lexical overlap such as "hospital wing" while
    never matching the object itself. Those windows are not good segment seeds
    for later frame sampling, so we drop them here.
    """
    component_scores = dict(alignment_rec.get("component_scores", {}))
    object_overlap = float(component_scores.get("object_overlap", 0.0))
    lexical_overlap = float(component_scores.get("lexical_overlap", 0.0))
    matched_subtitle = bool(alignment_rec.get("matched_subtitle_text"))
    if matched_subtitle and object_overlap <= 0.0 and lexical_overlap > 0.0:
        return False
    return True


def _initial_segment(
    *,
    alignment_index: int,
    alignment_rec: Dict[str, Any],
    candidate: Dict[str, Any],
    role_tags: Sequence[str],
    padding_sec: float,
) -> Dict[str, Any]:
    start_sec = max(0.0, float(alignment_rec["aligned_start_sec"]) - padding_sec)
    end_sec = max(start_sec, float(alignment_rec["aligned_end_sec"]) + padding_sec)
    component_scores = dict(alignment_rec.get("component_scores", {}))

    return {
        "segment_id": (
            f"cand{int(alignment_rec['candidate_index']):03d}_"
            f"scene{int(alignment_rec['scene_idx_non_omitted']):03d}_"
            f"{candidate.get('object_id', '')}"
        ),
        "candidate_index": int(alignment_rec["candidate_index"]),
        "object_id": candidate.get("object_id", ""),
        "canonical_name": candidate.get("canonical_name", ""),
        "candidate_source": candidate.get("candidate_source", ""),
        "possible_chekhov_score": float(candidate.get("possible_chekhov_score", 0.0)),
        "source_scene_indices": [int(alignment_rec["scene_idx_non_omitted"])],
        "source_alignment_indices": [alignment_index],
        "segment_role": _segment_role(role_tags),
        "segment_start_sec": round(start_sec, 3),
        "segment_end_sec": round(end_sec, 3),
        "priority_score": _priority_score(
            candidate_score=float(candidate.get("possible_chekhov_score", 0.0)),
            alignment_confidence=float(alignment_rec.get("confidence", 0.0)),
            role_tags=role_tags,
            candidate_source=candidate.get("candidate_source", ""),
        ),
        "alignment_confidence_max": float(alignment_rec.get("confidence", 0.0)),
        "reason_tags": _initial_reason_tags(
            candidate_source=candidate.get("candidate_source", ""),
            role_tags=role_tags,
            component_scores=component_scores,
        ),
        "matched_subtitle_texts": [alignment_rec.get("matched_subtitle_text", "")] if alignment_rec.get("matched_subtitle_text") else [],
        "matched_subtitle_indices": list(alignment_rec.get("matched_subtitle_indices", [])),
        "query_terms": list(alignment_rec.get("query_terms", [])),
    }


def _merge_reason_tags(left: Sequence[str], right: Sequence[str]) -> List[str]:
    out: List[str] = []
    for value in list(left) + list(right):
        if value not in out:
            out.append(value)
    return out


def _merge_segment_role(left: str, right: str) -> str:
    order = {
        "payoff": 3,
        "setup": 2,
        "reappearance": 1,
        "bridge": 0,
    }
    return left if order.get(left, 0) >= order.get(right, 0) else right


def _merge_segments(segments: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    first = dict(segments[0])
    first["segment_start_sec"] = min(float(seg["segment_start_sec"]) for seg in segments)
    first["segment_end_sec"] = max(float(seg["segment_end_sec"]) for seg in segments)
    first["source_scene_indices"] = sorted(
        {
            idx
            for seg in segments
            for idx in seg.get("source_scene_indices", [])
        }
    )
    first["source_alignment_indices"] = sorted(
        {
            idx
            for seg in segments
            for idx in seg.get("source_alignment_indices", [])
        }
    )
    first["priority_score"] = round(
        max(float(seg["priority_score"]) for seg in segments),
        4,
    )
    first["alignment_confidence_max"] = round(
        max(float(seg["alignment_confidence_max"]) for seg in segments),
        4,
    )
    first["segment_role"] = segments[0]["segment_role"]
    first["reason_tags"] = []
    first["matched_subtitle_texts"] = []
    first["matched_subtitle_indices"] = []
    first["query_terms"] = []

    for seg in segments:
        first["segment_role"] = _merge_segment_role(first["segment_role"], seg["segment_role"])
        first["reason_tags"] = _merge_reason_tags(first["reason_tags"], seg.get("reason_tags", []))
        for txt in seg.get("matched_subtitle_texts", []):
            if txt and txt not in first["matched_subtitle_texts"]:
                first["matched_subtitle_texts"].append(txt)
        for idx in seg.get("matched_subtitle_indices", []):
            if idx not in first["matched_subtitle_indices"]:
                first["matched_subtitle_indices"].append(idx)
        for term in seg.get("query_terms", []):
            if term not in first["query_terms"]:
                first["query_terms"].append(term)

    first["segment_id"] = (
        f"cand{first['candidate_index']:03d}_"
        f"{first['object_id']}_"
        f"seg{first['source_scene_indices'][0]:03d}"
    )
    first["segment_start_sec"] = round(float(first["segment_start_sec"]), 3)
    first["segment_end_sec"] = round(float(first["segment_end_sec"]), 3)
    return first


def _merge_nearby_segments(
    segments: Sequence[Dict[str, Any]],
    *,
    merge_gap_sec: float,
) -> List[Dict[str, Any]]:
    if not segments:
        return []

    ordered = sorted(
        segments,
        key=lambda seg: (seg["segment_start_sec"], seg["segment_end_sec"]),
    )
    groups: List[List[Dict[str, Any]]] = [[ordered[0]]]
    for seg in ordered[1:]:
        cur_group = groups[-1]
        cur_end = max(float(item["segment_end_sec"]) for item in cur_group)
        if float(seg["segment_start_sec"]) <= cur_end + merge_gap_sec:
            cur_group.append(seg)
        else:
            groups.append([seg])

    return [_merge_segments(group) for group in groups]


def build_candidate_segments(
    candidates_json: Dict[str, Any],
    alignment_json: Dict[str, Any],
    *,
    min_alignment_confidence: float = 0.15,
    padding_sec: float = 12.0,
    merge_gap_sec: float = 8.0,
    max_segments_per_object: int = 4,
) -> Dict[str, Any]:
    """
    Build candidate video segments from coarse alignment output.

    Args:
        candidates_json: Output from `chekhov_candidates.json`.
        alignment_json: Output from the coarse alignment layer.
        min_alignment_confidence: Minimum alignment confidence for a scene step
            to be retained as a candidate segment seed.
        padding_sec: Temporal padding added on both sides of each aligned window.
        merge_gap_sec: Maximum temporal gap allowed when merging nearby windows
            into a single candidate segment.
        max_segments_per_object: Hard cap on retained segments per object after
            ranking.

    Returns:
        JSON-serializable candidate segment object.
    """
    candidates_by_index = _candidate_lookup(candidates_json)
    if not candidates_by_index:
        raise ValueError("candidates_json contains no candidates")

    alignments = list(alignment_json.get("alignments", []))
    if not alignments:
        raise ValueError("alignment_json contains no alignments")

    seeded_segments_by_object: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    skipped_non_object_specific = 0
    for alignment_index, alignment_rec in enumerate(alignments):
        candidate_index = int(alignment_rec["candidate_index"])
        candidate = candidates_by_index.get(candidate_index)
        if candidate is None:
            continue

        confidence = float(alignment_rec.get("confidence", 0.0))
        if confidence < min_alignment_confidence:
            continue
        if not _has_object_specific_alignment(alignment_rec):
            skipped_non_object_specific += 1
            continue

        role_map = _scene_role_map(candidate)
        scene_idx = int(alignment_rec["scene_idx_non_omitted"])
        role_tags = role_map.get(scene_idx, [])
        segment = _initial_segment(
            alignment_index=alignment_index,
            alignment_rec=alignment_rec,
            candidate=candidate,
            role_tags=role_tags,
            padding_sec=padding_sec,
        )
        seeded_segments_by_object[segment["object_id"]].append(segment)

    final_segments: List[Dict[str, Any]] = []
    for object_id, seeded_segments in seeded_segments_by_object.items():
        merged = _merge_nearby_segments(
            seeded_segments,
            merge_gap_sec=merge_gap_sec,
        )
        merged.sort(
            key=lambda seg: (
                float(seg["priority_score"]),
                float(seg["alignment_confidence_max"]),
                -float(seg["segment_start_sec"]),
            ),
            reverse=True,
        )
        final_segments.extend(merged[:max_segments_per_object])

    final_segments.sort(
        key=lambda seg: (
            seg["object_id"],
            -float(seg["priority_score"]),
            float(seg["segment_start_sec"]),
        )
    )

    return {
        "meta": {
            "segments_returned": len(final_segments),
            "objects_considered": len(seeded_segments_by_object),
            "min_alignment_confidence": min_alignment_confidence,
            "padding_sec": padding_sec,
            "merge_gap_sec": merge_gap_sec,
            "max_segments_per_object": max_segments_per_object,
            "skipped_non_object_specific_alignments": skipped_non_object_specific,
            "build_type": "task3_candidate_segment_filtering",
        },
        "segments": final_segments,
    }
