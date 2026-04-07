# src/tracking_project/llm/curator_openai.py
import hashlib
import json
import os
from typing import Dict, List, Optional

from tracking_project.llm.prompts import build_curator_prompt

REQUIRED_KEYS = [
    "canonical_name","keep","physical_visualizable","category",
    "state_change_potential","narrative_action_centrality",
    "visual_grounded_score","confidence","short_rationale"
]

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def load_cache(cache_dir: str, key: str) -> Optional[dict]:
    path = os.path.join(cache_dir, f"{key}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def save_cache(cache_dir: str, key: str, obj: dict) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{key}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def openai_curate_one(model: str, entity: str, contexts: List[str], cache_dir: str) -> Dict:
    payload = {"entity": entity, "contexts": contexts[:8]}
    cache_key = _sha1(json.dumps(payload, sort_keys=True))
    cached = load_cache(cache_dir, cache_key)
    if cached is not None:
        return cached

    from openai import OpenAI
    client = OpenAI()

    prompt = build_curator_prompt(entity, contexts)

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "canonical_name": {"type": "string", "minLength": 1},
            "keep": {"type": "boolean"},
            "physical_visualizable": {"type": "boolean"},
            "category": {
                "type": "string",
                "enum": ["object","artifact","weapon","creature","vehicle","location","body_part","expression","abstract","person","other"]
            },
            "state_change_potential": {"type": "number", "minimum": 0, "maximum": 1},
            "narrative_action_centrality": {"type": "number", "minimum": 0, "maximum": 1},
            "visual_grounded_score": {"type": "number", "minimum": 0, "maximum": 1},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "short_rationale": {"type": "string", "maxLength": 120},
        },
        "required": REQUIRED_KEYS,
    }

    resp = client.responses.create(
        model=model,
        input=[{"role": "user", "content": prompt}],
        text={
            "format": {
                "type": "json_schema",
                "name": "curation",
                "strict": True,
                "schema": schema,
            }
        },
    )

    # With structured outputs, output_text should be valid JSON
    obj = json.loads(resp.output_text)

    save_cache(cache_dir, cache_key, obj)
    return obj