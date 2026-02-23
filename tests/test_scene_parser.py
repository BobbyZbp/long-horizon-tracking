from tracking_project.io.pdf_scene_parser import parse_pdf_scenes


class FakePage:
    def __init__(self, text: str):
        self._text = text

    def extract_text(self):
        return self._text


class FakePDF:
    def __init__(self, pages):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def fake_pdf_open(_path):
    # Two pages, multiple scene headings, includes INT/EXT with spaces
    p1 = FakePage(
        "\n".join([
            "1 INT. KITCHEN - DAY 1",
            "Action line one.",
            "",
            "2 EXT. STREET - NIGHT 2",
            "More action.",
            "",
            "3 EXT. /INT. CAR - NIGHT 3",
            "Driving action.",
        ])
    )
    p2 = FakePage(
        "\n".join([
            "4 OMITTED 4",
            "",
            "5 INT/EXT. FOREST - DAY 5",
            "Forest action.",
        ])
    )
    return FakePDF([p1, p2])


def test_scene_count_and_order():
    scenes, meta = parse_pdf_scenes("dummy.pdf", debug=True, pdf_open=fake_pdf_open)
    assert meta["scene_markers_total"] == 5
    assert meta["scene_markers_non_omitted"] == 4
    assert scenes[0].scene_idx == 0
    assert scenes[-1].scene_idx == 4


def test_marker_normalization():
    scenes, _ = parse_pdf_scenes("dummy.pdf", pdf_open=fake_pdf_open)
    assert scenes[0].marker_type == "INT."
    assert scenes[1].marker_type == "EXT."
    assert scenes[2].marker_type == "EXT/INT."
    assert scenes[3].marker_type == "OMITTED"
    assert scenes[4].marker_type == "INT/EXT."


def test_non_omitted_indexing():
    scenes, _ = parse_pdf_scenes("dummy.pdf", pdf_open=fake_pdf_open)

    # Non-omitted scenes get dense indices 0..3
    assert scenes[0].scene_idx_non_omitted == 0
    assert scenes[1].scene_idx_non_omitted == 1
    assert scenes[2].scene_idx_non_omitted == 2

    # OMITTED -> -1
    assert scenes[3].scene_idx_non_omitted == -1

    # Next non-omitted continues indexing
    assert scenes[4].scene_idx_non_omitted == 3