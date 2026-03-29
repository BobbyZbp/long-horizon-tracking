from typing import List

def build_curator_prompt(entity: str, contexts: List[str]) -> str:
    return (
        "You are a STRICT dataset curator for VIDEO-GROUNDED narrative tracking research.\n\n"

        "We want INSTANCE-LEVEL, TRACKABLE entities that:\n"
        "- Have persistent identity across scenes\n"
        "- Can be localized in video\n"
        "- Can re-enter narrative action after long absence\n\n"

        "Reject entities that are:\n"
        "- Generic background objects (roof, wall, bed, door, table)\n"
        "- Scene structural parts (floor, ceiling, window, corner)\n"
        "- Body parts or facial expressions\n"
        "- Actions, gestures, or events\n"
        "- Abstract concepts or time references\n"
        "- Groups of people or pronouns\n\n"
  
        "Prefer entities that are:\n"
        "- Named artifacts (e.g., Tom Riddle's diary)\n"
        "- Unique vehicles or creatures\n"
        "- Distinct magical objects or tools\n"
        "- Objects that undergo state change or drive action\n\n"

        "Return ONLY strict JSON with keys:\n"
        "canonical_name, keep, physical_visualizable, category,\n"
        "state_change_potential, narrative_action_centrality,\n"
        "visual_grounded_score, confidence, short_rationale.\n\n"

        f"Entity: {entity}\n"
        "Contexts:\n" + "\n".join(f"- {c}" for c in contexts[:8])
    )