"""
stage1_mine_llm.py

This module implements the full end-to-end pipeline for discovering
long-gap object reappearances in a screenplay.

Overview
--------
Given a screenplay PDF, this stage:

1) Parses scene structure.
2) Extracts physical entities per scene using an LLM.
3) Aggregates mentions across scenes by normalized entity key.
4) Computes temporal re-occurrence statistics.
5) Filters entities by semantic and structural thresholds.
6) Ranks and outputs high-gap "re-occurrence events".

Core Definitions:
------------
For each entity e appearing in non-omitted scenes:

    Let S(e) = sorted scene indices where e appears.

    gap_jump(e) = max_i ( S[i+1] - S[i] )

We interpret large gap_jump values as narrative disappearance
followed by meaningful re-activation.

- Deterministic given cached LLM outputs.
- Fully explainable: output includes field specification.
- Threshold-controlled: explicit filtering criteria.
This module defines:
    - AggregatedEntity: cross-scene statistics container.
    - mine_stage1_llm(): full pipeline execution.
    - _build_field_spec(): self-documenting output schema.
"""
from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from tracking_project.io.pdf_scene_parser import parse_pdf_scenes
from tracking_project.io.jsonio import safe_write_json
from tracking_project.text.cleaners import clean_scene_lines
from tracking_project.text.blocks import split_blocks, block_type
from tracking_project.llm.scene_extractor_openai import openai_scene_extract_entities
from tracking_project.scoring.ranking import (
    final_sort_key,
    gap_level_from_quantiles,
    quantile,
)


def _normalize_entity_key(name: str) -> str:
    """
    Normalize an entity canonical name into a stable key.

    This function is used to merge slightly different surface canonical names
    (e.g. "the diary" vs "Tom Riddle's diary") into a single entity bucket.

    Normalization steps:
      - Lowercase.
      - Strip leading/trailing whitespace.
      - Remove leading articles 'the', 'a', 'an'.
      - Remove most punctuation except internal whitespace and apostrophes.

    Args:
        name: Canonical name as produced by the LLM.

    Returns:
        A normalized string key suitable for aggregation.
    """
    import re

    t = name.strip().lower()
    t = re.sub(r"^(the|a|an)\s+", "", t)
    t = re.sub(r"[^\w\s']+", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


@dataclass
class AggregatedEntity:
    """
    Aggregated statistics for a single entity key across all scenes.

    Fields:
        entity_key: Normalized entity key used as dictionary key.
        canonical_names: Counter of canonical_name variants across scenes.
        categories: Counter of semantic categories (object / artifact / ...).
        surface_forms: Counter of surface forms across scenes.
        scenes_non_omitted: Sorted list of non-omitted scene indices where the
            entity appears at least once.
        per_scene_evidence: Mapping from scene index -> list of evidence snippets.
        centrality_scores: Per-mention narrative_action_centrality scores.
        state_scores: Per-mention state_change_potential scores.
        visual_scores: Per-mention visual_grounded_score scores.
        confidence_scores: Per-mention confidence scores.
    """
    entity_key: str
    canonical_names: Counter = field(default_factory=Counter)
    categories: Counter = field(default_factory=Counter)
    surface_forms: Counter = field(default_factory=Counter)
    scenes_non_omitted: List[int] = field(default_factory=list)
    per_scene_evidence: Dict[int, List[str]] = field(default_factory=lambda: defaultdict(list))
    centrality_scores: List[float] = field(default_factory=list)
    state_scores: List[float] = field(default_factory=list)
    visual_scores: List[float] = field(default_factory=list)
    confidence_scores: List[float] = field(default_factory=list)

    def register_mention(
        self,
        scene_idx_non_omitted: int,
        canonical_name: str,
        category: str,
        surface_forms: List[str],
        evidence: List[str],
        centrality: float,
        state_change: float,
        visual: float,
        confidence: float,
    ) -> None:
        """
        Update aggregated statistics with one LLM-extracted mention of this entity.

        Args:
            scene_idx_non_omitted: Non-omitted scene index where the entity appears.
            canonical_name: Canonical name from LLM for this mention.
            category: Category string from LLM.
            surface_forms: Surface forms for this entity in this scene.
            evidence: Evidence snippets for this mention.
            centrality: Narrative action centrality score in [0,1].
            state_change: State change potential score in [0,1].
            visual: Visual grounded score in [0,1].
            confidence: Confidence score in [0,1].
        """
        self.canonical_names[canonical_name] += 1
        self.categories[category] += 1
        for sf in surface_forms:
            self.surface_forms[sf] += 1

        if scene_idx_non_omitted not in self.scenes_non_omitted:
            self.scenes_non_omitted.append(scene_idx_non_omitted)
            self.scenes_non_omitted.sort()

        # Keep evidence per scene, but truncate to avoid unbounded growth
        if evidence:
            existing = self.per_scene_evidence[scene_idx_non_omitted]
            for snip in evidence:
                if snip not in existing and len(existing) < 5:
                    existing.append(snip)

        self.centrality_scores.append(float(centrality))
        self.state_scores.append(float(state_change))
        self.visual_scores.append(float(visual))
        self.confidence_scores.append(float(confidence))

    def aggregated_scores(self) -> Tuple[float, float, float, float]:
        """
        Compute mean scores over all mentions for this entity.

        Returns:
            A tuple (centrality_mean, visual_mean, state_mean, confidence_mean),
            each value in [0,1] or 0.0 if there were no mentions (which should not happen).
        """
        import math

        def _mean(xs: List[float]) -> float:
            return float(sum(xs) / len(xs)) if xs else 0.0

        centrality_mean = _mean(self.centrality_scores)
        visual_mean = _mean(self.visual_scores)
        state_mean = _mean(self.state_scores)
        confidence_mean = _mean(self.confidence_scores)
        # Clamp to [0,1] just in case
        return (
            min(max(centrality_mean, 0.0), 1.0),
            min(max(visual_mean, 0.0), 1.0),
            min(max(state_mean, 0.0), 1.0),
            min(max(confidence_mean, 0.0), 1.0),
        )


def mine_stage1_llm(
    *,
    pdf: str,
    out: str,
    llm_model: str = "gpt-4o-mini",
    cache_dir: str = "data/llm_cache",
    top_k: int = 100,
    min_gap: int = 20,
    keep_categories: List[str] | None = None,
    min_conf: float = 0.70,
    min_visual: float = 0.60,
    min_action: float = 0.45,
    min_state: float = 0.25,
    debug_scene_parse: bool = False,
    dump_scenes_path: str = "data/processed/scene_markers.json",
) -> Dict[str, Any]:
    """
    Stage-1 pipeline: LLM-first narrative-aware re-occurrence mining.

    High-level steps:
      1) Parse screenplay PDF into scenes.
      2) For each scene:
           - Clean lines and extract action text.
           - Call an LLM once to extract entities and per-mention scores.
      3) Aggregate mentions across scenes by normalized entity key.
      4) Compute per-entity:
           - scene indices where it appears
           - max jump gap between adjacent mentions (non-omitted index)
           - frequency and coverage
           - mean scores from LLM.
      5) Filter entities by gap + score thresholds and category whitelist.
      6) Rank remaining entities and output top_k events as JSON.

    This function is deterministic given:
      - the screenplay PDF
      - the LLM model and API behavior
      - the cache directory contents.

    Args:
        pdf: Path to the screenplay PDF file.
        out: Path to the final events JSON file.
        llm_model: OpenAI model name to use for scene-level extraction.
        cache_dir: Root directory for LLM cache files.
        top_k: Maximum number of events to return in the final output.
        min_gap: Minimum required gap_jump (in non-omitted scene index) for an
            entity to be considered interesting.
        keep_categories: List of allowed categories as returned by the LLM
            (e.g. ["object","artifact","weapon","creature","vehicle"]).
        min_conf: Minimum mean confidence score [0,1] to keep an entity.
        min_visual: Minimum mean visual grounded score [0,1].
        min_action: Minimum mean narrative action centrality [0,1].
        min_state: Minimum mean state change potential [0,1].
        debug_scene_parse: If True, pdf_scene_parser will also record near-miss
            candidate headings for debugging undercounting.
        dump_scenes_path: Optional path where a compact representation of scene
            markers is dumped for inspection.

    Returns:
        A dict representing the final output JSON object, also written to disk.
    """
    keep_categories = keep_categories or ["object", "artifact", "weapon", "creature", "vehicle"]

    t0 = time.time()
    print("[phase] parse scenes...", flush=True)
    scenes, scene_meta = parse_pdf_scenes(pdf, debug=debug_scene_parse, dump_path=dump_scenes_path)
    non_omitted = scene_meta["scene_markers_non_omitted"]

    if non_omitted < 50:
        print(f"[warn] scenes(non-omitted)={non_omitted} seems low; please inspect {dump_scenes_path}", flush=True)

    # Map from non-omitted scene index -> Scene object for later lookup.
    scene_by_non_omitted: Dict[int, Any] = {
        s.scene_idx_non_omitted: s for s in scenes if s.scene_idx_non_omitted >= 0
    }

    print("[phase] LLM scene extraction...", flush=True)

    entities_by_key: Dict[str, AggregatedEntity] = {}
    total_scene_calls = 0

    # Iterate over scenes in non-omitted order for stable progress reporting
    scenes_iter = [s for s in scenes if s.scene_idx_non_omitted >= 0]
    scenes_iter.sort(key=lambda s: s.scene_idx_non_omitted)

    for sc in tqdm(scenes_iter, desc="scene-llm"):
        clean_lines = clean_scene_lines(sc.lines)
        blocks = split_blocks(clean_lines)

        # Concatenate action blocks into a single string
        action_chunks: List[str] = []
        for blk in blocks:
            if block_type(blk) == "action":
                txt = " ".join(ln.strip() for ln in blk if ln.strip())
                if txt:
                    action_chunks.append(txt)
        action_text = "\n".join(action_chunks).strip()

        if not action_text:
            continue

        total_scene_calls += 1
        entities = openai_scene_extract_entities(
            model=llm_model,
            scene_idx_non_omitted=sc.scene_idx_non_omitted,
            heading=sc.heading,
            action_text=action_text,
            cache_root=cache_dir,
        )

        for e in entities:
            # Basic sanity checks + hard filtering at mention level
            if not e.get("keep", False):
                continue
            if not e.get("physical_visualizable", False):
                continue

            category = e.get("category", "other")
            if category not in keep_categories:
                continue

            try:
                canonical_name = str(e["canonical_name"])
                surface_forms = list(e.get("surface_forms", []))
                evidence = list(e.get("evidence", []))
                centrality = float(e.get("narrative_action_centrality", 0.0))
                state_change = float(e.get("state_change_potential", 0.0))
                visual = float(e.get("visual_grounded_score", 0.0))
                confidence = float(e.get("confidence", 0.0))
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[warn] skipping entity due to malformed fields: {e} ({exc})")
                continue

            key = _normalize_entity_key(canonical_name)
            if not key:
                continue

            if key not in entities_by_key:
                entities_by_key[key] = AggregatedEntity(entity_key=key)

            agg = entities_by_key[key]
            agg.register_mention(
                scene_idx_non_omitted=sc.scene_idx_non_omitted,
                canonical_name=canonical_name,
                category=category,
                surface_forms=surface_forms,
                evidence=evidence,
                centrality=centrality,
                state_change=state_change,
                visual=visual,
                confidence=confidence,
            )

    print(f"[info] scene_llm_calls={total_scene_calls}", flush=True)

    print("[phase] aggregate + compute gaps...", flush=True)
    candidates: List[Dict[str, Any]] = []

    for key, agg in entities_by_key.items():
        scenes_sorted = sorted(set(agg.scenes_non_omitted))
        if len(scenes_sorted) < 2:
            continue

        # Compute max gap between adjacent mentions
        max_gap = -1
        best_pair: Tuple[int, int] | None = None
        for a, b in zip(scenes_sorted, scenes_sorted[1:]):
            gap = b - a
            if gap > max_gap:
                max_gap = gap
                best_pair = (a, b)
        if best_pair is None:
            continue

        gap_jump = max_gap
        gap_absent = max(gap_jump - 1, 0)
        if gap_jump < min_gap:
            continue

        freq_scenes = len(scenes_sorted)
        span = max(scenes_sorted) - min(scenes_sorted) + 1
        coverage = float(freq_scenes) / float(span) if span > 0 else 0.0

        centrality_mean, visual_mean, state_mean, conf_mean = agg.aggregated_scores()

        # Apply entity-level score thresholds
        if conf_mean < min_conf:
            continue
        if visual_mean < min_visual:
            continue
        if centrality_mean < min_action:
            continue
        if state_mean < min_state:
            continue

        # Choose majority canonical_name and category
        canonical_name_final = (agg.canonical_names.most_common(1)[0][0]
                                if agg.canonical_names else key)
        category_final = (agg.categories.most_common(1)[0][0]
                          if agg.categories else "object")

        scene_a_idx, scene_b_idx = best_pair
        sa = scene_by_non_omitted.get(scene_a_idx)
        sb = scene_by_non_omitted.get(scene_b_idx)
        if sa is None or sb is None:
            continue

        # Context A/B from per-scene evidence
        context_a = list(agg.per_scene_evidence.get(scene_a_idx, []))[:5]
        context_b = list(agg.per_scene_evidence.get(scene_b_idx, []))[:5]

        # Surface forms: keep the top few
        surface_top = [sf for sf, _ in agg.surface_forms.most_common(5)]

        candidates.append(
            {
                "entity_key": key,
                "canonical_name": canonical_name_final,
                "entity_type": category_final,
                "surface_forms": surface_top,
                "scene_a_idx_non_omitted": scene_a_idx,
                "scene_b_idx_non_omitted": scene_b_idx,
                "gap_jump": gap_jump,
                "gap_absent": gap_absent,
                "freq_scenes": freq_scenes,
                "coverage": coverage,
                "scene_a_label_raw": sa.scene_label_raw,
                "scene_b_label_raw": sb.scene_label_raw,
                "page_a": sa.page_start,
                "page_b": sb.page_start,
                "heading_a": sa.heading,
                "heading_b": sb.heading,
                "context_a": context_a,
                "context_b": context_b,
                "scores": {
                    "narrative_action_centrality": centrality_mean,
                    "visual_grounded_score": visual_mean,
                    "state_change_potential": state_mean,
                    "confidence": conf_mean,
                },
                "provenance": {
                    "scenes_non_omitted_list": scenes_sorted,
                },
            }
        )

    # Compute difficulty bins based on gap_jump quantiles
    gaps = [c["gap_jump"] for c in candidates]
    q50 = quantile(gaps, 0.50) if gaps else 0.0
    q75 = quantile(gaps, 0.75) if gaps else 0.0

    for c in candidates:
        c["difficulty"] = {
            "gap_level": gap_level_from_quantiles(c["gap_jump"], q50, q75)
        }

    # Final ranking
    candidates.sort(
        key=lambda c: final_sort_key(
            c["gap_jump"],
            c["scores"]["narrative_action_centrality"],
            c["scores"]["visual_grounded_score"],
            c["scores"]["state_change_potential"],
        ),
        reverse=True,
    )
    events = candidates[:top_k]

    # Build field spec for meta (for self-documentation inside JSON)
    field_spec = _build_field_spec()

    out_obj = {
        "meta": {
            "status": "complete",
            "pdf": pdf,
            "scenes_non_omitted": non_omitted,
            "total_entities_aggregated": len(entities_by_key),
            "events_returned": len(events),
            "gap_quantiles": {"q50": q50, "q75": q75},
            "gap_definition": "gap_jump(e) = max_i (s_{i+1}-s_i) over non-omitted scene indices",
            "field_spec": field_spec,
            "elapsed_sec": round(time.time() - t0, 2),
        },
        "events": events,
    }

    safe_write_json(out, out_obj)
    print(f"[ok] stage1 curated={len(events)} -> {out}", flush=True)
    return out_obj


def _build_field_spec() -> Dict[str, Any]:
    """
    Build a self-contained specification for the fields in the output JSON.

    The spec is included in meta['field_spec'] so that anyone inspecting the
    output file can see how each field is defined and computed.

    Returns:
        A nested dict describing the meaning and computation of each top-level
        'events[*]' field.
    """
    return {
        "entity_key": {
            "type": "string",
            "description": (
                "Normalized entity key used for aggregation. "
                "Computed as lowercased canonical_name with leading articles "
                "removed and punctuation stripped."
            ),
            "computation": "entity_key = normalize(canonical_name)",
        },
        "canonical_name": {
            "type": "string",
            "description": (
                "Most frequent canonical_name string returned by the LLM for this "
                "entity across scenes."
            ),
            "computation": "argmax over canonical_names counter",
        },
        "entity_type": {
            "type": "string",
            "description": (
                "Most frequent category assigned by LLM across mentions. One of "
                "{object, artifact, weapon, creature, vehicle, location, other}."
            ),
            "computation": "argmax over categories counter",
        },
        "surface_forms": {
            "type": "list[string]",
            "description": (
                "Most frequent surface forms used for this entity across scenes "
                "in the screenplay."
            ),
            "computation": "top-k from surface_forms counter",
        },
        "scene_a_idx_non_omitted": {
            "type": "int",
            "description": (
                "Non-omitted scene index of the earlier mention in the max-gap pair."
            ),
            "computation": "s_A = s_i where s_{i+1}-s_i is maximal",
        },
        "scene_b_idx_non_omitted": {
            "type": "int",
            "description": (
                "Non-omitted scene index of the later mention in the max-gap pair."
            ),
            "computation": "s_B = s_{i+1} where s_{i+1}-s_i is maximal",
        },
        "gap_jump": {
            "type": "int",
            "description": (
                "Max jump in non-omitted scene indices between adjacent mentions "
                "of the entity."
            ),
            "computation": "gap_jump(e) = max_i (s_{i+1} - s_i)",
        },
        "gap_absent": {
            "type": "int",
            "description": (
                "Number of scenes strictly between the two mentions that realize "
                "gap_jump. Zero if gap_jump <= 1."
            ),
            "computation": "gap_absent(e) = max(gap_jump(e) - 1, 0)",
        },
        "freq_scenes": {
            "type": "int",
            "description": "Number of distinct non-omitted scenes in which the entity appears.",
            "computation": "freq_scenes(e) = |{s_1,...,s_n}|",
        },
        "coverage": {
            "type": "float",
            "description": (
                "Fraction of scenes in the span from first to last mention that "
                "actually mention the entity."
            ),
            "computation": "coverage(e) = freq_scenes / (s_n - s_1 + 1)",
        },
        "scene_a_label_raw": {
            "type": "string",
            "description": "Raw label of scene A as printed in the script (e.g. '12A').",
        },
        "scene_b_label_raw": {
            "type": "string",
            "description": "Raw label of scene B as printed in the script (e.g. '45').",
        },
        "page_a": {
            "type": "int",
            "description": "PDF page number where scene A begins.",
        },
        "page_b": {
            "type": "int",
            "description": "PDF page number where scene B begins.",
        },
        "heading_a": {
            "type": "string",
            "description": "Scene heading line for scene A.",
        },
        "heading_b": {
            "type": "string",
            "description": "Scene heading line for scene B.",
        },
        "context_a": {
            "type": "list[string]",
            "description": (
                "Short evidence snippets for this entity taken from scene A "
                "action text, supplied by the LLM."
            ),
        },
        "context_b": {
            "type": "list[string]",
            "description": (
                "Short evidence snippets for this entity taken from scene B "
                "action text, supplied by the LLM."
            ),
        },
        "scores": {
            "type": "object",
            "description": (
                "Per-entity mean scores in [0,1] derived from LLM per-mention scores."
            ),
            "fields": {
                "narrative_action_centrality": {
                    "type": "float",
                    "description": (
                        "How central the entity is to actions across scenes "
                        "(0=background,1=drives action)."
                    ),
                    "computation": "mean over mention-level narrative_action_centrality scores",
                },
                "visual_grounded_score": {
                    "type": "float",
                    "description": (
                        "How clearly the entity can be grounded in visual perception."
                    ),
                    "computation": "mean over mention-level visual_grounded_score scores",
                },
                "state_change_potential": {
                    "type": "float",
                    "description": (
                        "How likely the entity is to undergo meaningful state changes."
                    ),
                    "computation": "mean over mention-level state_change_potential scores",
                },
                "confidence": {
                    "type": "float",
                    "description": "LLM's mean confidence in its assessments.",
                    "computation": "mean over mention-level confidence scores",
                },
            },
        },
        "difficulty": {
            "type": "object",
            "description": (
                "Difficulty bin derived from gap_jump empirical quantiles "
                "over all curated entities."
            ),
            "fields": {
                "gap_level": {
                    "type": "string",
                    "description": "One of {'medium','long','extreme'} based on gap_jump.",
                    "computation": (
                        "if gap_jump <= q50 -> 'medium'; "
                        "elif gap_jump <= q75 -> 'long'; "
                        "else 'extreme'."
                    ),
                }
            },
        },
        "provenance": {
            "type": "object",
            "description": (
                "Auxiliary fields describing where this entity was observed "
                "in the screenplay."
            ),
            "fields": {
                "scenes_non_omitted_list": {
                    "type": "list[int]",
                    "description": "Sorted list of all non-omitted scene indices mentioning the entity.",
                }
            },
        },
    }