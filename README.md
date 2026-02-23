# Stage 1: Narrative-Aware Re-occurrence Mining (Screenplay → Events)

This repo implements **Stage 1** of a tracking project: automatically mine **narrative re-occurrence entities**
from a screenplay (PDF). We focus on **physical, visually-groundable entities** (artifacts, creatures, vehicles, etc.)
that **disappear for a while** and then **re-occur** later in the narrative.

The output is a structured `events.json` dataset with:
- canonical entity names + surface forms
- max-jump gap between adjacent mentions (scene-level)
- evidence snippets (context) for the two scenes defining the max gap
- semantic scores (action centrality / visual groundedness / state-change / confidence)
- a built-in `field_spec` describing how each field is computed

---

## What is “re-occurrence” here?

We parse the screenplay into ordered non-omitted scenes:
\[
s \in \{0,1,\dots,N-1\}
\]

For an entity with scene indices:
\[
s_1 < s_2 < \dots < s_n
\]

We define:

- **gap_jump**:
\[
\text{gap\_jump} = \max_i (s_{i+1} - s_i)
\]

- **gap_absent**:
\[
\text{gap\_absent} = \max(\text{gap\_jump} - 1, 0)
\]

The event uses the pair \((s_A, s_B)\) that realizes `gap_jump`.

---

## Pipeline Overview

1) **Parse scenes** (deterministic + unit-tested)  
- Input: screenplay PDF  
- Output: a list of `Scene` objects with `scene_idx_non_omitted`, heading, and lines.

2) **Clean + extract action text**  
- Remove layout noise (CONTINUED, Rev., headers).  
- Split into blocks; keep **action blocks** and concatenate into `action_text`.

3) **LLM scene extraction (cached + JSON schema)**  
For each scene, call the LLM once to extract:
- physical/visualizable entities only
- category and semantic scores
- 1–3 evidence snippets per entity

All per-scene LLM outputs are cached in `data/llm_cache/scene_extract/`.

4) **Aggregate entities + compute gaps**  
- Merge mentions by normalized entity key.
- Compute `gap_jump` and select max-gap pair.
- Attach `context_a` and `context_b` from evidence snippets.

5) **Filter + rank + export**  
- Keep only allowed categories and minimum score thresholds.
- Rank by a deterministic score combining gap and semantic scores.
- Export `events.json` with a self-contained `field_spec`.

---

## Installation

Create a Python environment (>=3.10):

```bash
pip install -e ".[dev]"
pip install -e ".[pipeline]"