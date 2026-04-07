# Long-Horizon Narrative-to-Video Tracking

This repository contains the code for a screenplay-to-video pipeline that
surfaces likely narrative objects, aligns them to movie time, narrows them into
object-centric candidate segments, and prepares review assets for downstream
tracking experiments.

The repo is now organized around the workflow we actually use:

1. screenplay parsing and Stage-1 long-gap mining
2. high-recall scene inventory
3. climax scoring
4. cross-scene Chekhov-style candidate validation
5. coarse subtitle-backed video alignment
6. candidate segment filtering
7. default frame sampling or manual object-centric refinement
8. optional SAM3 notebook experiments on prepared clips/frames

## Source Of Truth

The tracked source-of-truth in this repo is:

- code under `src/`
- runnable entrypoints under `scripts/pipeline/`, `scripts/refine/`, and `scripts/archive/`
- tests under `tests/`
- documentation under `README.md` and `docs/`
- the original `sam3_video_segmentation.ipynb` notebook

The following are intentionally treated as local-only and are not part of the
canonical tracked repo surface:

- `data/` contents
- local virtual environments
- a local `sam3/` checkout
- local SAM3 notebook variants
- generated clips, sampled frames, caches, and contact sheets

This means the recommended cluster workflow is:

1. clone the repo
2. install the code environment
3. copy the needed local data onto the machine separately
4. run the pipeline or notebook from the cloned codebase

## Pipeline Outputs

The main JSON artifacts produced by the pipeline are:

- `run1.json`
  - Stage-1 long-gap event prior
- `scene_inventory.json`
  - high-recall per-scene object inventory
- `scene_climax.json`
  - scene-level climax scores
- `chekhov_candidates.json`
  - validated cross-scene narrative candidates
- `aligned_candidates.json`
  - coarse script-to-video alignment windows
- `candidate_segments.json`
  - object-centric candidate segments for review or refinement

In practice, we often reuse existing upstream JSON files and rerun only the
later stages we are actively debugging.

## Scripts Layout

The `scripts/` folder is organized by role:

- `scripts/pipeline/`
  - canonical batch pipeline entrypoints
- `scripts/refine/`
  - object-centric refinement tools for manual review and SAM3 preparation
- `scripts/archive/`
  - historical or inspection utilities that are not part of the canonical path

## Installation

Python 3.10+ is required.

### Recommended development / cluster setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### What `requirements.txt` installs

`requirements.txt` is the convenience environment for:

- the canonical pipeline
- tests
- refinement/video helpers

It does **not** install:

- a local SAM3 checkout
- SAM3 model weights
- movie files or subtitle assets

### Optional extras

If you want only the package and optional extras directly:

```bash
pip install -e .
pip install -e .[dev,video]
```

If you want the archived spaCy-based baseline tools as well:

```bash
pip install -e .[legacy]
python -m spacy download en_core_web_sm
```

## Environment Variables

Common environment variables used by the repo:

- `OPENAI_API_KEY`
  - required for LLM-backed extraction/scoring stages
- `OPENAI_MODEL`
  - optional default model override for scripts that support it
- `HF_TOKEN`
  - only needed for SAM3 notebook/model download workflows

## Ubuntu / Cluster Workflow

Typical cluster flow:

```bash
git clone <repo-url>
cd tracking-project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then copy local data onto the machine separately, for example:

- screenplay PDFs
- subtitles
- local JSON artifacts under `data/`
- short review clips and frame directories if you already prepared them

The repo intentionally does not assume those files will come from Git.

## Notes On SAM3

The repo keeps the original notebook:

- `sam3_video_segmentation.ipynb`

But SAM3 itself is treated as an external dependency/workspace concern:

- local `sam3/` checkouts are ignored
- local SAM3 notebook variants are ignored
- model checkpoints are not stored in this repo

## Historical Material

These remain useful as reference, but they are not the main repo entrypoint:

- `docs/task1_pipeline_spec.md`
- `TASK2_ALIGNMENT_REPORT.md`
- `TASK34_FILTER_SAMPLING_REPORT.md`

They describe earlier stages and design decisions, while this README reflects
the current repo surface.
