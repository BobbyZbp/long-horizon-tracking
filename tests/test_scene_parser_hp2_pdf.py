import os
import pytest
from tracking_project.io.pdf_scene_parser import parse_pdf_scenes

HP2 = "data/raw/harry-potter-and-the-chamber-of-secrets-2002.pdf"

@pytest.mark.skipif(not os.path.exists(HP2), reason="HP2 PDF not found in data/raw/")
def test_hp2_scene_count():
    scenes, meta = parse_pdf_scenes(HP2, debug=True)
    # What we validated empirically:
    assert meta["scene_markers_non_omitted"] == 116
    assert meta["scene_markers_total"] >= 140
    assert meta["near_miss_count"] == 0
    # sanity: first/last idx are consistent
    assert scenes[0].scene_idx == 0
    assert scenes[-1].scene_idx == meta["scene_markers_total"] - 1