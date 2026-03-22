"""
test_frame_sampling.py

Unit tests for frame sampling.
"""
from __future__ import annotations

from tracking_project.pipeline.frame_sampling import sample_candidate_frames


def test_sample_candidate_frames_copies_uniform_samples(tmp_path):
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    for idx in range(0, 6):
        (frame_dir / f"frame_{idx:06d}.jpg").write_bytes(f"frame-{idx}".encode("utf-8"))

    segments_json = {
        "segments": [
            {
                "segment_id": "seg_a",
                "object_id": "diary",
                "canonical_name": "Tom Riddle's diary",
                "segment_start_sec": 0.0,
                "segment_end_sec": 2.0,
            }
        ]
    }

    out = sample_candidate_frames(
        segments_json,
        frame_dir=str(frame_dir),
        fps=2.0,
        out_frames_dir=str(tmp_path / "sampled"),
        frame_index_base=0,
        frames_per_segment=3,
        strategy="uniform",
    )

    assert out["meta"]["segments_sampled"] == 1
    sampled = out["segments"][0]["sampled_frames"]
    assert [rec["frame_index"] for rec in sampled] == [0, 2, 4]
    for rec in sampled:
        assert (tmp_path / "sampled" / "seg_a" / f"sample_{rec['sample_index']:02d}_f{rec['frame_index']:06d}.jpg").exists()


def test_sample_candidate_frames_uses_nearest_available_frame(tmp_path):
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    for idx in [0, 4]:
        (frame_dir / f"frame_{idx:06d}.jpg").write_bytes(f"frame-{idx}".encode("utf-8"))

    segments_json = {
        "segments": [
            {
                "segment_id": "seg_b",
                "object_id": "locket",
                "canonical_name": "locket",
                "segment_start_sec": 1.0,
                "segment_end_sec": 1.0,
            }
        ]
    }

    out = sample_candidate_frames(
        segments_json,
        frame_dir=str(frame_dir),
        fps=2.0,
        out_frames_dir=str(tmp_path / "sampled"),
        frame_index_base=0,
        frames_per_segment=1,
        strategy="uniform",
    )

    sampled = out["segments"][0]["sampled_frames"][0]
    assert sampled["target_frame_index"] == 2
    assert sampled["frame_index"] == 0
