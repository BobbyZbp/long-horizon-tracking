"""
frame_sampling.py

This module implements deterministic frame sampling from candidate video
segments.

Why this file exists
--------------------
The candidate segment filtering layer produces candidate time segments, but
those segments are still temporal
intervals. Before running any tracker, the pipeline needs a visual inspection
layer that converts segment timestamps into concrete frame files.

Pipeline position
-----------------
- prior layer: candidate segment filtering
- current layer: frame sampling
- next layer: SAM-based tracking demo

This module:
    - indexes an extracted frame directory
    - converts segment timestamps into frame indices
    - selects representative sample times deterministically
    - copies the chosen frames into per-segment output folders
    - writes a manifest that records the exact timestamp/frame mapping

The output of this module is the canonical sampled-frame manifest.
"""
from __future__ import annotations

import os
import re
import shutil
from bisect import bisect_left
from typing import Any, Dict, Iterable, List, Sequence, Tuple


_FRAME_INDEX_RE = re.compile(r"(\d+)")
_DEFAULT_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _parse_frame_index_from_name(filename: str) -> int | None:
    stem, _ext = os.path.splitext(os.path.basename(filename))
    matches = _FRAME_INDEX_RE.findall(stem)
    if not matches:
        return None
    return int(matches[-1])


def _frame_lookup(frame_dir: str) -> Dict[int, str]:
    lookup: Dict[int, str] = {}
    for name in sorted(os.listdir(frame_dir)):
        path = os.path.join(frame_dir, name)
        if not os.path.isfile(path):
            continue
        _stem, ext = os.path.splitext(name)
        if ext.lower() not in _DEFAULT_EXTS:
            continue
        frame_index = _parse_frame_index_from_name(name)
        if frame_index is None:
            continue
        lookup[frame_index] = path
    return lookup


def _nearest_available_index(target: int, available_indices: Sequence[int]) -> int:
    pos = bisect_left(available_indices, target)
    if pos <= 0:
        return available_indices[0]
    if pos >= len(available_indices):
        return available_indices[-1]

    left = available_indices[pos - 1]
    right = available_indices[pos]
    return left if abs(left - target) <= abs(right - target) else right


def _sample_times(
    start_sec: float,
    end_sec: float,
    *,
    frames_per_segment: int,
    strategy: str,
) -> List[float]:
    if frames_per_segment <= 0:
        raise ValueError("frames_per_segment must be positive")
    if end_sec < start_sec:
        raise ValueError("segment end must be >= segment start")

    duration = end_sec - start_sec
    if frames_per_segment == 1:
        return [start_sec + duration / 2.0]

    if strategy == "uniform":
        step = duration / float(frames_per_segment - 1) if frames_per_segment > 1 else 0.0
        return [start_sec + i * step for i in range(frames_per_segment)]
    if strategy == "first_middle_last":
        times = [start_sec, start_sec + duration / 2.0, end_sec]
        while len(times) < frames_per_segment:
            times.append(start_sec + duration / 2.0)
        return times[:frames_per_segment]

    raise ValueError(f"Unsupported sampling strategy: {strategy}")


def _timestamp_to_frame_index(
    timestamp_sec: float,
    *,
    fps: float,
    frame_index_base: int,
) -> int:
    return int(round(timestamp_sec * fps)) + frame_index_base


def sample_candidate_frames(
    segments_json: Dict[str, Any],
    *,
    frame_dir: str,
    fps: float,
    out_frames_dir: str,
    frame_index_base: int = 0,
    frames_per_segment: int = 6,
    strategy: str = "uniform",
) -> Dict[str, Any]:
    """
    Build sampled frame outputs from candidate segments.

    Args:
        segments_json: Output from the candidate segment filtering layer.
        frame_dir: Directory containing extracted movie frames.
        fps: Frames per second for the frame extraction source.
        out_frames_dir: Root output directory for copied sample frames.
        frame_index_base: Index base used in the frame filenames.
        frames_per_segment: Number of representative frames to sample.
        strategy: Sampling strategy. One of {"uniform", "first_middle_last"}.

    Returns:
        JSON-serializable sampled-frame manifest.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    if strategy not in {"uniform", "first_middle_last"}:
        raise ValueError(f"Unsupported sampling strategy: {strategy}")

    segments = list(segments_json.get("segments", []))
    if not segments:
        raise ValueError("segments_json contains no segments")

    lookup = _frame_lookup(frame_dir)
    if not lookup:
        raise ValueError(f"No indexed image frames found in: {frame_dir}")

    available_indices = sorted(lookup)
    os.makedirs(out_frames_dir, exist_ok=True)

    manifest_segments: List[Dict[str, Any]] = []
    for segment in segments:
        segment_id = str(segment["segment_id"])
        start_sec = float(segment["segment_start_sec"])
        end_sec = float(segment["segment_end_sec"])
        sample_times = _sample_times(
            start_sec,
            end_sec,
            frames_per_segment=frames_per_segment,
            strategy=strategy,
        )

        segment_out_dir = os.path.join(out_frames_dir, segment_id)
        os.makedirs(segment_out_dir, exist_ok=True)

        sampled_frames: List[Dict[str, Any]] = []
        for sample_idx, sample_time in enumerate(sample_times):
            target_index = _timestamp_to_frame_index(
                sample_time,
                fps=fps,
                frame_index_base=frame_index_base,
            )
            chosen_index = _nearest_available_index(target_index, available_indices)
            source_path = lookup[chosen_index]
            source_name = os.path.basename(source_path)
            ext = os.path.splitext(source_name)[1].lower()
            output_name = f"sample_{sample_idx:02d}_f{chosen_index:06d}{ext}"
            output_path = os.path.join(segment_out_dir, output_name)
            shutil.copy2(source_path, output_path)

            sampled_frames.append(
                {
                    "sample_index": sample_idx,
                    "timestamp_sec": round(sample_time, 3),
                    "target_frame_index": target_index,
                    "frame_index": chosen_index,
                    "source_path": source_path,
                    "output_path": output_path,
                }
            )

        manifest_segments.append(
            {
                "segment_id": segment_id,
                "object_id": segment.get("object_id", ""),
                "canonical_name": segment.get("canonical_name", ""),
                "segment_start_sec": start_sec,
                "segment_end_sec": end_sec,
                "fps": fps,
                "frame_index_base": frame_index_base,
                "sampling_strategy": strategy,
                "sampled_frames": sampled_frames,
            }
        )

    return {
        "meta": {
            "segments_sampled": len(manifest_segments),
            "frame_dir": frame_dir,
            "fps": fps,
            "frame_index_base": frame_index_base,
            "frames_per_segment": frames_per_segment,
            "sampling_strategy": strategy,
            "build_type": "task4_frame_sampling",
        },
        "segments": manifest_segments,
    }
