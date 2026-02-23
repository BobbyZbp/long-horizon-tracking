"""
prompts_scene.py

This module defines the scene-level LLM interface for Stage-1 narrative-aware
entity mining.

It contains:

1) build_scene_extraction_prompt(...)
   - Constructs the natural-language instruction given to the LLM.
   - Defines the semantic task (what to extract, what to ignore, how to score).

2) scene_entity_schema()
   - Defines a strict JSON Schema used with Structured Outputs.
   - Enforces well-formed, typed, and bounded LLM output.

Architectural role
------------------
This module is the semantic contract between:
    - Deterministic scene parsing (structure layer)
    - LLM-based entity extraction (semantic layer)

It is the ONLY place where we:
    - Define what counts as a trackable entity
    - Define scoring semantics (centrality, state change, visual groundedness)
    - Restrict output format

All downstream aggregation, ranking, and gap computation rely on the
correctness and consistency of this contract.

It purely defines the LLM task specification.
"""
from __future__ import annotations

from typing import Dict


def build_scene_extraction_prompt(
    scene_idx_non_omitted: int,
    heading: str,
    action_text: str,
) -> str:
    """
    Build the natural-language prompt used for scene-level entity extraction.

    The goal of this prompt is to:
      - Give the model enough context to understand what happens in the scene.
      - Instruct it to output only PHYSICAL, VISUALLY-GROUNDED entities that
        can plausibly be tracked in video.
      - Ask for per-entity semantic scores (0..1) and short textual evidence
        snippets that justify the extraction.

    Args:
        scene_idx_non_omitted: Sequential non-omitted scene index for this scene.
        heading: Scene heading (e.g., "INT. GREAT HALL - NIGHT").
        action_text: Concatenated action-block text for the scene.

    Returns:
        A single prompt string to pass to the OpenAI Responses API.
    """
    return (
        "You are an expert screenplay annotator working on VIDEO-GROUNDED research.\n"
        "Your task is to read the action description of ONE scene from a film, and "
        "extract PHYSICAL entities that can be visually tracked in the film.\n\n"
        "For this scene:\n"
        f"- scene_idx_non_omitted = {scene_idx_non_omitted}\n"
        f"- heading = {heading}\n\n"
        "Consider ONLY the ACTION description (no character psychology). "
        "Focus on objects, artifacts, weapons, creatures, vehicles and similar.\n\n"
        "For each entity you output:\n"
        "- canonical_name: a short, unambiguous name (e.g. 'Tom Riddle's diary', 'Flying Ford Anglia').\n"
        "- surface_forms: ALL distinct ways this entity is referred to in THIS scene "
        "(e.g. ['diary', \"Riddle's diary\"]).\n"
        "- category: one of {object, artifact, weapon, creature, vehicle, location, other}.\n"
        "- physical_visualizable: true if it can be seen or localized in a film frame.\n"
        "- narrative_action_centrality: a score in [0,1] describing how central this "
        "entity is to the actions in this scene (0=background, 1=drives the action).\n"
        "- state_change_potential: a score in [0,1] describing whether the entity can "
        "undergo meaningful state changes (e.g. broken, opened, moved, destroyed).\n"
        "- visual_grounded_score: a score in [0,1] describing how clearly this entity "
        "could be grounded in visual perception.\n"
        "- confidence: your confidence in your overall assessment for this entity.\n"
        "- keep: true if you believe this entity is a good candidate for long-term "
        "object tracking across scenes; false otherwise.\n"
        "- evidence: 1-3 short text snippets (taken from the input text, or very close "
        "paraphrases) that show the entity being used or acted upon.\n\n"
        "DO NOT output:\n"
        "- Abstract concepts (e.g. 'memory', 'hope').\n"
        "- Purely psychological states (e.g. 'fear', 'anger').\n"
        "- Pure camera directions or verbs (e.g. 'OPEN', 'CUT TO', 'REVERSE').\n"
        "- People (characters). We handle characters separately.\n\n"
        "Return a SINGLE valid JSON object with:\n"
        "{\n"
        '  \"scene_idx_non_omitted\": <int>,\n'
        '  \"entities\": [\n'
        "    {\n"
        '      \"canonical_name\": <str>,\n'
        '      \"surface_forms\": [<str>, ...],\n'
        '      \"category\": <str>,\n'
        '      \"physical_visualizable\": <bool>,\n'
        '      \"narrative_action_centrality\": <float in [0,1]>,\n'
        '      \"state_change_potential\": <float in [0,1]>,\n'
        '      \"visual_grounded_score\": <float in [0,1]>,\n'
        '      \"confidence\": <float in [0,1]>,\n'
        '      \"keep\": <bool>,\n'
        '      \"evidence\": [<str>, ...]\n'
        "    },\n"
        "    ...\n"
        "  ]\n"
        "}\n\n"
        "Now process the following scene action text:\n"
        "----- ACTION TEXT START -----\n"
        f"{action_text}\n"
        "----- ACTION TEXT END -----\n"
    )


def scene_entity_schema() -> Dict:
    """
    JSON schema for scene-level entity extraction.

    This is used with the OpenAI Responses API Structured Outputs to ensure that
    the model returns strictly valid JSON that matches the expected structure.

    Returns:
        A Python dict representing a JSON Schema with:
          - scene_idx_non_omitted: integer
          - entities: array of entity objects as described in
            build_scene_extraction_prompt().
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "scene_idx_non_omitted": {"type": "integer"},
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "canonical_name": {"type": "string", "minLength": 1},
                        "surface_forms": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1},
                        },
                        "category": {
                            "type": "string",
                            "enum": [
                                "object",
                                "artifact",
                                "weapon",
                                "creature",
                                "vehicle",
                                "location",
                                "other",
                            ],
                        },
                        "physical_visualizable": {"type": "boolean"},
                        "narrative_action_centrality": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                        "state_change_potential": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                        "visual_grounded_score": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                        "keep": {"type": "boolean"},
                        "evidence": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1},
                        },
                    },
                    "required": [
                        "canonical_name",
                        "surface_forms",
                        "category",
                        "physical_visualizable",
                        "narrative_action_centrality",
                        "state_change_potential",
                        "visual_grounded_score",
                        "confidence",
                        "keep",
                        "evidence",
                    ],
                },
            },
        },
        "required": ["scene_idx_non_omitted", "entities"],
    }