"""
video_alignment.py

This module performs coarse script-to-video alignment.

Alignment goal
--------------
Given narrative candidates and subtitle timestamps, produce approximate video
time windows for the scenes in each candidate's loose narrative chain.

Why this module exists
----------------------
The first video-facing alignment pass does not need frame-accurate localization
yet. It needs believable, debuggable candidate windows that can later support:
    - candidate segment filtering
    - frame sampling
    - SAM-based tracking checks

Design principles
-----------------
- Coarse before precise: use story order as the primary prior.
- Dialogue-aware: use subtitle text overlap as a secondary refinement signal.
- Explainable: keep component scores and matched subtitle text.
- Deterministic: no LLM calls or stochastic search.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence, Set

from tracking_project.io.subtitles import SubtitleSegment


_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "day",
    "ext",
    "for",
    "from",
    "in",
    "int",
    "into",
    "later",
    "moment",
    "moments",
    "night",
    "of",
    "on",
    "room",
    "scene",
    "the",
    "to",
}


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9']+", text.lower())
    return [tok for tok in tokens if tok not in _STOPWORDS and len(tok) > 1]


def _overlap_score(query_terms: Set[str], text_terms: Set[str]) -> float:
    if not query_terms or not text_terms:
        return 0.0
    return len(query_terms & text_terms) / float(len(query_terms))


def _overlap_terms(query_terms: Set[str], text_terms: Set[str]) -> Set[str]:
    if not query_terms or not text_terms:
        return set()
    return query_terms & text_terms


def _subtitle_windows(
    segments: Sequence[SubtitleSegment],
    *,
    neighborhood: int,
) -> List[Dict[str, Any]]:
    windows: List[Dict[str, Any]] = []
    for idx in range(len(segments)):
        left = max(0, idx - neighborhood)
        right = min(len(segments), idx + neighborhood + 1)
        chunk = segments[left:right]
        text = " ".join(seg.text for seg in chunk)
        windows.append(
            {
                "center_index": idx,
                "segment_indices": [seg.index for seg in chunk],
                "start_sec": chunk[0].start_sec,
                "end_sec": chunk[-1].end_sec,
                "center_sec": segments[idx].center_sec,
                "text": text,
                "tokens": set(_tokenize(text)),
            }
        )
    return windows


def _scene_query_terms(candidate: Dict[str, Any], scene_rec: Dict[str, Any]) -> Dict[str, Set[str]]:
    general_terms: Set[str] = set()
    object_terms: Set[str] = set()

    general_terms.update(_tokenize(candidate.get("canonical_name", "")))
    general_terms.update(_tokenize(scene_rec.get("heading", "")))

    for obj_name in scene_rec.get("matched_inventory_objects", []):
        tokens = set(_tokenize(obj_name))
        general_terms.update(tokens)
        object_terms.update(tokens)

    # Full evidence snippets often contain broader narrative context that can
    # hijack matching toward the wrong object. Only keep evidence terms that
    # reinforce the object-specific query rather than every story token.
    evidence_terms: Set[str] = set()
    for ev in scene_rec.get("inventory_evidence", []):
        evidence_terms.update(_tokenize(ev))

    if not object_terms:
        object_terms.update(_tokenize(candidate.get("canonical_name", "")))

    general_terms.update(term for term in evidence_terms if term in object_terms)

    return {
        "general_terms": general_terms,
        "object_terms": object_terms,
    }


def _fallback_window(
    *,
    expected_time_sec: float,
    movie_duration_sec: float,
    default_window_sec: float,
) -> Dict[str, Any]:
    half = default_window_sec / 2.0
    start_sec = max(0.0, expected_time_sec - half)
    end_sec = min(movie_duration_sec, expected_time_sec + half)
    return {
        "segment_indices": [],
        "start_sec": start_sec,
        "end_sec": end_sec,
        "center_sec": 0.5 * (start_sec + end_sec),
        "text": "",
    }


def build_script_video_alignment(
    candidates_json: Dict[str, Any],
    scene_inventory_json: Dict[str, Any],
    subtitle_segments: Sequence[SubtitleSegment],
    *,
    search_radius_sec: float = 420.0,
    default_window_sec: float = 75.0,
    neighborhood: int = 1,
) -> Dict[str, Any]:
    """
    Build coarse subtitle-backed alignment windows for narrative candidates.

    Args:
        candidates_json: Output from `chekhov_candidates.json`.
        scene_inventory_json: Scene inventory JSON, used for total scene count.
        subtitle_segments: Ordered subtitle segments with timestamps.
        search_radius_sec: Temporal radius around the expected scene time.
        default_window_sec: Fallback window size when subtitle evidence is weak.
        neighborhood: Number of subtitle neighbors to include on each side when
            building subtitle windows.

    Returns:
        JSON-serializable alignment object with `meta` and `alignments`.
    """
    if not subtitle_segments:
        raise ValueError("subtitle_segments is empty")

    total_scenes = int(scene_inventory_json.get("meta", {}).get("scenes_non_omitted", 0))
    if total_scenes <= 0:
        raise ValueError("scene_inventory_json.meta.scenes_non_omitted must be positive")

    movie_duration_sec = max(seg.end_sec for seg in subtitle_segments)
    subtitle_windows = _subtitle_windows(subtitle_segments, neighborhood=neighborhood)

    alignments: List[Dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates_json.get("candidates", [])):
        scene_chain = sorted(
            candidate.get("loose_scene_chain", []),
            key=lambda rec: rec["scene_idx_non_omitted"],
        )
        ranked_windows_by_scene: List[List[Dict[str, Any]]] = []

        for scene_rec in scene_chain:
            scene_idx = int(scene_rec["scene_idx_non_omitted"])
            expected_time_sec = (
                scene_idx / float(max(total_scenes - 1, 1))
            ) * movie_duration_sec
            query = _scene_query_terms(candidate, scene_rec)

            candidates_for_scene: List[Dict[str, Any]] = []
            for window in subtitle_windows:
                delta = abs(window["center_sec"] - expected_time_sec)
                if delta > search_radius_sec:
                    continue

                order_score = 1.0 - (delta / float(max(search_radius_sec, 1.0)))
                shared_general = _overlap_terms(query["general_terms"], window["tokens"])
                shared_object = _overlap_terms(query["object_terms"], window["tokens"])
                lexical_overlap = _overlap_score(query["general_terms"], window["tokens"])
                object_overlap = _overlap_score(query["object_terms"], window["tokens"])

                # Avoid pretending that a nearby but unrelated subtitle is
                # evidence for the object. If there is no object-specific hit,
                # require at least modest general overlap; otherwise fall back
                # to the expected-time window.
                if not shared_object and len(shared_general) < 2:
                    continue

                confidence = (
                    0.45 * order_score
                    + 0.20 * lexical_overlap
                    + 0.35 * object_overlap
                )
                candidates_for_scene.append(
                    {
                        "window": window,
                        "expected_time_sec": expected_time_sec,
                        "confidence": round(confidence, 4),
                        "order_score": round(order_score, 4),
                        "lexical_overlap": round(lexical_overlap, 4),
                        "object_overlap": round(object_overlap, 4),
                        "query_terms": sorted(query["general_terms"]),
                    }
                )

            if not candidates_for_scene:
                fallback = _fallback_window(
                    expected_time_sec=expected_time_sec,
                    movie_duration_sec=movie_duration_sec,
                    default_window_sec=default_window_sec,
                )
                candidates_for_scene = [
                    {
                        "window": fallback,
                        "expected_time_sec": expected_time_sec,
                        "confidence": 0.0,
                        "order_score": 0.0,
                        "lexical_overlap": 0.0,
                        "object_overlap": 0.0,
                        "query_terms": sorted(query["general_terms"]),
                    }
                ]

            candidates_for_scene.sort(
                key=lambda rec: (
                    rec["confidence"],
                    rec["order_score"],
                    rec["lexical_overlap"],
                    rec["object_overlap"],
                ),
                reverse=True,
            )
            ranked_windows_by_scene.append(candidates_for_scene)

        prev_center_sec = -1.0
        for scene_rec, ranked_windows in zip(scene_chain, ranked_windows_by_scene):
            chosen = ranked_windows[0]
            for option in ranked_windows:
                if option["window"]["center_sec"] + 1e-6 >= prev_center_sec:
                    chosen = option
                    break

            prev_center_sec = chosen["window"]["center_sec"]
            window = chosen["window"]
            alignments.append(
                {
                    "candidate_index": candidate_index,
                    "object_id": candidate.get("object_id", ""),
                    "canonical_name": candidate.get("canonical_name", ""),
                    "candidate_source": candidate.get("candidate_source", ""),
                    "scene_idx_non_omitted": int(scene_rec["scene_idx_non_omitted"]),
                    "heading": scene_rec.get("heading", ""),
                    "aligned_start_sec": round(float(window["start_sec"]), 3),
                    "aligned_end_sec": round(float(window["end_sec"]), 3),
                    "expected_time_sec": round(float(chosen["expected_time_sec"]), 3),
                    "confidence": float(chosen["confidence"]),
                    "matched_subtitle_indices": list(window.get("segment_indices", [])),
                    "matched_subtitle_text": window.get("text", ""),
                    "query_terms": list(chosen["query_terms"]),
                    "component_scores": {
                        "order_score": float(chosen["order_score"]),
                        "lexical_overlap": float(chosen["lexical_overlap"]),
                        "object_overlap": float(chosen["object_overlap"]),
                    },
                }
            )

    return {
        "meta": {
            "alignment_method": "coarse_order_plus_subtitle_overlap",
            "scene_candidates_considered": len(candidates_json.get("candidates", [])),
            "subtitle_segments": len(subtitle_segments),
            "subtitle_windows": len(subtitle_windows),
            "alignments_returned": len(alignments),
            "search_radius_sec": search_radius_sec,
            "default_window_sec": default_window_sec,
            "neighborhood": neighborhood,
            "movie_duration_sec": round(movie_duration_sec, 3),
        },
        "alignments": alignments,
    }
