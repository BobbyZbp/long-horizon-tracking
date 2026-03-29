from __future__ import annotations

import time
from collections import Counter, defaultdict
from typing import Any, Dict, List

from tqdm import tqdm

from tracking_project.io.pdf_scene_parser import parse_pdf_scenes
from tracking_project.io.jsonio import safe_write_json
from tracking_project.text.blocks import block_type, split_blocks
from tracking_project.text.extraction_spacy import extract_mentions_spacy, load_spacy
from tracking_project.llm.curator_openai import openai_curate_one
from tracking_project.scoring.ranking import final_sort_key, gap_level_from_quantiles, quantile


def mine_events_llm(
    *,
    pdf: str,
    out: str,
    spacy_model: str = "en_core_web_sm",
    llm_model: str = "gpt-4o-mini",
    cache_dir: str = "data/llm_cache",
    top_k: int = 100,
    min_gap: int = 20,          # MIN jump between adjacent mentions (in non-omitted scene index)
    preselect_k: int = 200,
    include_dialogue: bool = True,
    keep_categories: List[str] | None = None,
    min_conf: float = 0.75,
    min_visual: float = 0.70,
    min_action: float = 0.55,
    min_state: float = 0.35,
    debug_scene_parse: bool = False,
    dump_scenes_path: str = "data/processed/scene_markers.json",
    checkpoint_every: int = 1,
) -> Dict[str, Any]:
    """
    Stage-1 pipeline:
    - parse screenplay into scenes
    - spaCy extracts noun mentions + contexts
    - compute per-entity max jump between ADJACENT mentions (your definition)
    - LLM curates for physical/visualizable narrative objects
    - output top_k events with rich semantic context
    """

    keep_categories = keep_categories or ["object", "artifact", "weapon", "creature", "vehicle"]

    t0 = time.time()
    print("[phase] parse scenes...", flush=True)
    scenes, scene_meta = parse_pdf_scenes(pdf, debug=debug_scene_parse, dump_path=dump_scenes_path)

    non_omitted = scene_meta["scene_markers_non_omitted"]
    # For HP2 PDF, non_omitted should be around ~116
    if non_omitted < 105:
        print(f"[warn] scenes(non-omitted)={non_omitted} looks low; inspect {dump_scenes_path}", flush=True)

    scene_by_idx = {s.scene_idx: s for s in scenes}

    print("[phase] spacy extract...", flush=True)
    nlp = load_spacy(spacy_model)

    # entity -> scene_idx(all markers) -> aggregated mention info
    per_ent = defaultdict(lambda: defaultdict(lambda: {
        "contexts_action": [],
        "contexts_dialogue": [],
        "verbs_action": Counter(),
        "verbs_dialogue": Counter(),
        "co_action": Counter(),
        "co_dialogue": Counter(),
        "surface_forms": Counter(),
        "action_mentions": 0,
        "dialogue_mentions": 0,
    }))

    for sc in scenes:
        if sc.marker_type == "OMITTED":
            continue

        for blk in split_blocks(sc.lines):
            btype = block_type(blk)

            if btype == "dialogue":
                if not include_dialogue:
                    continue
                text = " ".join(blk[1:]).strip()
                if not text:
                    continue
                doc = nlp(text)
                recs = extract_mentions_spacy(doc, source="dialogue")
            else:
                text = " ".join(blk).strip()
                if not text:
                    continue
                doc = nlp(text)
                recs = extract_mentions_spacy(doc, source="action")

            for r in recs:
                ent = r["ent_norm"]
                slot = per_ent[ent][sc.scene_idx]
                slot["surface_forms"][r["ent_surface"]] += 1

                if r["source"] == "action":
                    slot["action_mentions"] += 1
                    if r["sent_text"] not in slot["contexts_action"]:
                        slot["contexts_action"].append(r["sent_text"])
                    for v in r["verbs"]:
                        slot["verbs_action"][v] += 1
                    for c in r["co_persons"]:
                        slot["co_action"][c] += 1
                else:
                    slot["dialogue_mentions"] += 1
                    if r["sent_text"] not in slot["contexts_dialogue"]:
                        slot["contexts_dialogue"].append(r["sent_text"])
                    for v in r["verbs"]:
                        slot["verbs_dialogue"][v] += 1
                    for c in r["co_persons"]:
                        slot["co_dialogue"][c] += 1

    print("[phase] rough candidates (max adjacent jump)...", flush=True)
    rough = []
    for ent, by_scene in per_ent.items():
        scene_idxs_all = sorted(by_scene.keys())
        if len(scene_idxs_all) < 2:
            continue

        # Require at least one action mention somewhere to avoid dialogue-only abstract junk
        action_scene_count = sum(1 for i in scene_idxs_all if by_scene[i]["action_mentions"] > 0)
        if action_scene_count < 1:
            continue

        # Compute max jump between ADJACENT mentions using non-omitted index
        max_jump = -1
        best_pair = None  # (a_idx_all, b_idx_all)
        for a_idx, b_idx in zip(scene_idxs_all, scene_idxs_all[1:]):
            sa = scene_by_idx[a_idx]
            sb = scene_by_idx[b_idx]
            ga = sa.scene_idx_non_omitted
            gb = sb.scene_idx_non_omitted
            if ga < 0 or gb < 0:
                continue
            jump = gb - ga  # <-- your definition
            if jump > max_jump:
                max_jump = jump
                best_pair = (a_idx, b_idx)

        if best_pair is None:
            continue
        if max_jump < min_gap:
            continue

        rough.append((max_jump, ent, best_pair[0], best_pair[1], scene_idxs_all, action_scene_count))

    rough.sort(key=lambda x: x[0], reverse=True)
    rough = rough[:preselect_k]
    print(f"[info] rough_preselect={len(rough)}", flush=True)

    def passes(llm: Dict[str, Any]) -> bool:
        return (
            llm.get("keep", False)
            and llm.get("confidence", 0.0) >= min_conf
            and llm.get("physical_visualizable", False)
            and llm.get("category", "other") in keep_categories
            and llm.get("visual_grounded_score", 0.0) >= min_visual
            and llm.get("narrative_action_centrality", 0.0) >= min_action
            and llm.get("state_change_potential", 0.0) >= min_state
        )

    curated: List[Dict[str, Any]] = []
    partial_path = out + ".partial.json"

    def checkpoint(processed: int):
        safe_write_json(partial_path, {
            "meta": {
                "status": "partial",
                "processed_candidates": processed,
                "elapsed_sec": round(time.time() - t0, 2),
            },
            "events": curated
        })

    print("[phase] LLM curate...", flush=True)
    try:
        for i, (max_jump, ent, a_idx_all, b_idx_all, scene_idxs_all, action_scene_count) in enumerate(tqdm(rough, desc="LLM-curate")):
            sa = scene_by_idx[a_idx_all]
            sb = scene_by_idx[b_idx_all]
            ga = sa.scene_idx_non_omitted
            gb = sb.scene_idx_non_omitted
            if ga < 0 or gb < 0:
                # Shouldn't happen (we skip OMITTED), but be safe
                continue

            slot_a = per_ent[ent][a_idx_all]
            slot_b = per_ent[ent][b_idx_all]

            ctx = slot_a["contexts_action"][:3] + slot_b["contexts_action"][:3]
            ctx += slot_a["contexts_dialogue"][:1] + slot_b["contexts_dialogue"][:1]
            ctx = [c for c in ctx if c]

            print(f"[{i+1}/{len(rough)}] entity='{ent}' gap_jump={max_jump} (non_omitted)", flush=True)
            llm = openai_curate_one(llm_model, ent, ctx, cache_dir=cache_dir)

            if not passes(llm):
                print(
                    f"    rejected cat={llm.get('category')} conf={llm.get('confidence',0):.2f} "
                    f"vis={llm.get('visual_grounded_score',0):.2f} act={llm.get('narrative_action_centrality',0):.2f} "
                    f"state={llm.get('state_change_potential',0):.2f} why='{llm.get('short_rationale','')}'",
                    flush=True
                )
                if checkpoint_every > 0 and (i + 1) % checkpoint_every == 0:
                    checkpoint(i + 1)
                continue

            surface = slot_a["surface_forms"].most_common(1)[0][0] if slot_a["surface_forms"] else ent

            event = {
                "entity": ent,
                "canonical_name": llm.get("canonical_name", ent),
                "entity_surface": surface,
                "entity_type": llm.get("category", "object"),
                "llm_curator": llm,

                # Scene identifiers for reproducibility
                "scene_a_idx_all": a_idx_all,
                "scene_b_idx_all": b_idx_all,
                "scene_a_idx_non_omitted": ga,
                "scene_b_idx_non_omitted": gb,

                # --- Gap definition (your choice) ---
                "gap_jump": max_jump,                 # = gb - ga
                "gap_absent": max(0, max_jump - 1),   # optional, derived

                "scene_a_label_raw": sa.scene_label_raw,
                "scene_b_label_raw": sb.scene_label_raw,
                "page_a": sa.page_start,
                "page_b": sb.page_start,
                "heading_a": sa.heading,
                "heading_b": sb.heading,

                # semantic context
                "context_a": slot_a["contexts_action"][:3] + slot_a["contexts_dialogue"][:2],
                "context_b": slot_b["contexts_action"][:3] + slot_b["contexts_dialogue"][:2],

                "salience": {
                    "freq_scenes_all": len(scene_idxs_all),
                    "action_scene_count_all": action_scene_count,
                    "scene_idxs_all_list": scene_idxs_all[:80],
                },

                "difficulty": {
                    "gap_level": None,  # filled after quantiles
                    "action_strength": float(llm.get("narrative_action_centrality", 0.0) * 5.0),
                    "visual_grounded_score": float(llm.get("visual_grounded_score", 0.0)),
                }
            }

            curated.append(event)
            print(f"    ACCEPT ✅ canonical='{event['canonical_name']}' cat={event['entity_type']}", flush=True)

            if checkpoint_every > 0 and (i + 1) % checkpoint_every == 0:
                checkpoint(i + 1)
            if len(curated) >= top_k:
                break

    except KeyboardInterrupt:
        print("\n[interrupt] Ctrl+C: writing partial checkpoint...", flush=True)
        checkpoint(i + 1 if "i" in locals() else 0)
        print(f"[interrupt] saved {partial_path}", flush=True)
        return {"meta": {"status": "partial"}, "events": curated}

    # finalize difficulty bins + final ranking
    gaps = [e["gap_jump"] for e in curated]
    q50 = quantile(gaps, 0.50) if gaps else 0.0
    q75 = quantile(gaps, 0.75) if gaps else 0.0
    for e in curated:
        e["difficulty"]["gap_level"] = gap_level_from_quantiles(e["gap_jump"], q50, q75)

    # Final sorting uses gap_jump (your definition)
    curated.sort(
        key=lambda e: final_sort_key(
            e["gap_jump"],
            e["llm_curator"].get("narrative_action_centrality", 0.0),
            e["llm_curator"].get("visual_grounded_score", 0.0),
            e["llm_curator"].get("state_change_potential", 0.0),
        ),
        reverse=True
    )
    curated = curated[:top_k]

    out_obj = {
        "meta": {
            "status": "complete",
            "pdf": pdf,
            "scenes_non_omitted": non_omitted,
            "rough_preselect": len(rough),
            "returned_events": len(curated),
            "gap_quantiles": {"q50": q50, "q75": q75},
            "elapsed_sec": round(time.time() - t0, 2),
            "gap_definition": "max adjacent jump on scene_idx_non_omitted: gap_jump = s[i+1]-s[i]",
        },
        "events": curated
    }
    safe_write_json(out, out_obj)
    safe_write_json(partial_path, {"meta": {"status": "final_checkpoint"}, "events": curated})
    print(f"[ok] curated={len(curated)} -> {out}", flush=True)
    return out_obj