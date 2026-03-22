from __future__ import annotations
import json

from tracking_project.io.pdf_scene_parser import parse_pdf_scenes
from tracking_project.llm.prompts_scene import build_climax_scoring_prompt
from tracking_project.llm.scene_extractor_openai import call_llm_json


def load_events(path):

    with open(path) as f:
        return json.load(f)


def build_scene_object_map(events):

    scene_objects = {}

    for event in events["events"]:

        name = event["canonical_name"]

        for s in event["provenance"]["scenes_non_omitted_list"]:

            if s not in scene_objects:
                scene_objects[s] = []

            if name not in scene_objects[s]:
                scene_objects[s].append(name)

    return scene_objects


def mine_climax_llm(pdf, events, out, llm_model):

    scenes, scene_meta = parse_pdf_scenes(pdf)

    # only keep real scenes
    scenes = [s for s in scenes if s.scene_idx_non_omitted >= 0]

    events_json = load_events(events)

    scene_objects = build_scene_object_map(events_json)

    story_summary = (
        "Harry Potter returns to Hogwarts where mysterious attacks petrify students. "
        "The Chamber of Secrets is reopened and Harry eventually confronts Tom Riddle "
        "and the Basilisk."
    )

    result = {"scenes": {}}

    for scene in scenes:

        action_text = "\n".join(scene.lines)

        prompt = build_climax_scoring_prompt(
            story_summary,
            scene.scene_idx_non_omitted,
            scene.heading,
            action_text,
        )

        response = call_llm_json(prompt, model=llm_model)

        score = response["climax_score"]

        result["scenes"][scene.scene_idx_non_omitted] = {
            "heading": scene.heading,
            "climax_score": score,
            "objects": scene_objects.get(scene.scene_idx_non_omitted, []),
        }

        print(f"Scene {scene.scene_idx_non_omitted} → {score}")

    with open(out, "w") as f:
        json.dump(result, f, indent=2)