"""
scene_extractor_openai.py

This module executes scene-level entity extraction using the OpenAI Responses API.

It is the execution boundary between:
    - Deterministic narrative preprocessing (scene parsing, block splitting)
    - LLM-based semantic annotation (entity extraction with scores)

Responsibilities
----------------
For each scene:

1) Build the LLM prompt using build_scene_extraction_prompt().
2) Enforce strict structured output using scene_entity_schema().
3) Cache results using a stable SHA-1 key derived from:
       (model, scene_idx_non_omitted, heading, action_text)
4) Return parsed entity dictionaries for downstream aggregation.

- Deterministic: identical inputs → identical cache key → no re-query.
- Isolated: this is the ONLY file that touches the OpenAI API.
- Schema-enforced: output format guaranteed by Structured Outputs.
- Reproducible: cached JSON files make experiments stable and re-usable.

This module contains no aggregation, ranking, or filtering logic.
It strictly performs scene-level semantic extraction.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional

from tracking_project.llm.prompts_scene import (
    build_scene_extraction_prompt,
    scene_entity_schema,
)


def _sha1(s: str) -> str:
    """
    Compute a short SHA-1 hash of a string.

    This is used to build stable cache keys for scene-level LLM calls, so we
    never pay twice for the same (model, input) pair.
    """
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _cache_dir_scene(cache_root: str) -> str:
    """
    Resolve the subdirectory for scene-extraction cache files.

    Args:
        cache_root: Root directory for all LLM caches.

    Returns:
        A path like '<cache_root>/scene_extract'.
    """
    path = os.path.join(cache_root, "scene_extract")
    os.makedirs(path, exist_ok=True)
    return path


def _load_scene_cache(cache_root: str, key: str) -> Optional[Dict[str, Any]]:
    """
    Load a cached scene-extraction result if it exists.

    Args:
        cache_root: Root directory for LLM caches.
        key: SHA-1 hex string identifying this (model, scene, text) combination.

    Returns:
        Parsed JSON object from cache, or None if no cache entry exists.
    """
    path = os.path.join(_cache_dir_scene(cache_root), f"{key}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_scene_cache(cache_root: str, key: str, obj: Dict[str, Any]) -> None:
    """
    Save a scene-extraction result to cache.

    Args:
        cache_root: Root directory for LLM caches.
        key: SHA-1 hex string identifying the cache entry.
        obj: Parsed JSON object to be stored.
    """
    path = os.path.join(_cache_dir_scene(cache_root), f"{key}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def openai_scene_extract_entities(
    model: str,
    scene_idx_non_omitted: int,
    heading: str,
    action_text: str,
    cache_root: str,
) -> List[Dict[str, Any]]:
    """
    Run the LLM once for a single scene to extract entities and their scores.

    This function is the only place where we talk to the OpenAI API for
    scene-level extraction. It:
      - Builds a prompt using the scene index, heading, and action text.
      - Uses the Responses API with a JSON schema to enforce well-formed output.
      - Caches the result on disk to avoid repeat charges and ensure reproducibility.

    Args:
        model: OpenAI model name (e.g. 'gpt-4o-mini').
        scene_idx_non_omitted: Non-omitted scene index (0-based).
        heading: Scene heading (e.g. "INT. GREAT HALL - NIGHT").
        action_text: Concatenated action-block text for this scene. If empty,
            this function returns an empty list without calling the API.
        cache_root: Root directory for LLM cache files (e.g. 'data/llm_cache').

    Returns:
        A list of entity dicts, each matching the schema from
        tracking_project.llm.prompts_scene.scene_entity_schema()['properties']['entities']['items'].
        If action_text is empty, returns [].
    """
    if not action_text.strip():
        return []

    payload = {
        "model": model,
        "scene_idx_non_omitted": scene_idx_non_omitted,
        "heading": heading,
        "action_text": action_text,
    }
    cache_key = _sha1(json.dumps(payload, sort_keys=True))

    cached = _load_scene_cache(cache_root, cache_key)
    if cached is not None:
        return cached.get("entities", [])

    # Import OpenAI lazily to avoid a hard dependency for non-LLM tests
    from openai import OpenAI

    client = OpenAI()
    prompt = build_scene_extraction_prompt(scene_idx_non_omitted, heading, action_text)
    schema = scene_entity_schema()

    resp = client.responses.create(
        model=model,
        input=[{"role": "user", "content": prompt}],
        text={
            "format": {
                "type": "json_schema",
                "name": "scene_entities",
                "schema": schema,
                "strict": True,
            }
        },
    )

    # With structured outputs, output_text should be a valid JSON string
    obj = json.loads(resp.output_text)
    if not isinstance(obj, dict) or "entities" not in obj:
        raise RuntimeError(f"Scene extractor returned invalid JSON: {obj}")

    _save_scene_cache(cache_root, cache_key, obj)
    return obj.get("entities", [])