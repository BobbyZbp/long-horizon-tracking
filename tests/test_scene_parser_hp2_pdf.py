"""
test_scene_parser_hp2_pdf.py

Smoke test for the real HP2 screenplay PDF when the local environment has the
required PDF dependency installed.
"""
import importlib.util
import os

import pytest

from tracking_project.io.pdf_scene_parser import parse_pdf_scenes

HP2 = "data/raw/harry-potter-and-the-chamber-of-secrets-2002.pdf"
HAS_PDFPLUMBER = importlib.util.find_spec("pdfplumber") is not None

@pytest.mark.skipif(not os.path.exists(HP2), reason="HP2 PDF not found in data/raw/")
@pytest.mark.skipif(not HAS_PDFPLUMBER, reason="pdfplumber not installed in current interpreter")
def test_hp2_scene_count():
    scenes, meta = parse_pdf_scenes(HP2, debug=True)
    # What we validated empirically:
    assert meta["scene_markers_non_omitted"] == 116
    assert meta["scene_markers_total"] >= 140
    assert meta["near_miss_count"] == 0
    # sanity: first/last idx are consistent
    assert scenes[0].scene_idx == 0
    assert scenes[-1].scene_idx == meta["scene_markers_total"] - 1
