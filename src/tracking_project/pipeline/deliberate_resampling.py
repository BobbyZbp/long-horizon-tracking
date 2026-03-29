"""
deliberate_resampling.py

This module implements a manual-first, object-centric resampling workflow for
small pilot experiments.

Why this file exists
--------------------
The main candidate segment pipeline is optimized for scalable narrowing of the
search space. For short-horizon manual validation, that default behavior can
still be too scene-centric: a segment may land near the right event while
missing the actual object in sampled frames.

This module exists to support a deliberate resampling layer where we:
    - pick a specific high-confidence object segment
    - expand its window around the current best segment center
    - sample the window much more densely than the default pipeline
    - save frames and review artifacts for human inspection

Pipeline position
-----------------
- prior layer: candidate segment filtering
- current layer: deliberate object-centric resampling
- next layer: manual window confirmation and short-clip preparation for SAM3

This layer is intentionally narrow in scope. It is designed for a single
object or a tiny number of objects, not for whole-movie batch processing.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class SelectedSegment:
    """Normalized segment record for deliberate resampling."""

    segment_id: str
    object_id: str
    canonical_name: str
    segment_role: str
    priority_score: float
    alignment_confidence_max: float
    segment_start_sec: float
    segment_end_sec: float
    source_scene_indices: Sequence[int]


@dataclass(frozen=True)
class ExplicitWindow:
    """Normalized explicit time window for manual refinement."""

    start_sec: float
    end_sec: float


def _seconds_to_hhmmss(seconds: float) -> str:
    total = int(round(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _slugify(value: str) -> str:
    value = value.strip().lower().replace(" ", "_")
    out = []
    for ch in value:
        if ch.isalnum() or ch in {"_", "-"}:
            out.append(ch)
    return "".join(out) or "object"


def select_segment(
    segments_json: Dict[str, Any],
    *,
    object_id: str,
    segment_id: str | None = None,
) -> SelectedSegment:
    """
    Select a target segment from candidate segment output.

    If segment_id is provided, the matching segment is returned. Otherwise,
    the highest-priority segment for the object is selected.
    """
    segments = list(segments_json.get("segments", []))
    if not segments:
        raise ValueError("segments_json contains no segments")

    matches = [seg for seg in segments if str(seg.get("object_id", "")) == object_id]
    if segment_id is not None:
        matches = [seg for seg in matches if str(seg.get("segment_id", "")) == segment_id]

    if not matches:
        raise ValueError(
            f"No segment found for object_id={object_id!r}"
            + (f" and segment_id={segment_id!r}" if segment_id else "")
        )

    best = max(
        matches,
        key=lambda seg: (
            float(seg.get("priority_score", 0.0)),
            float(seg.get("alignment_confidence_max", 0.0)),
            -float(seg.get("segment_start_sec", 0.0)),
        ),
    )

    return SelectedSegment(
        segment_id=str(best["segment_id"]),
        object_id=str(best["object_id"]),
        canonical_name=str(best.get("canonical_name", best["object_id"])),
        segment_role=str(best.get("segment_role", "")),
        priority_score=float(best.get("priority_score", 0.0)),
        alignment_confidence_max=float(best.get("alignment_confidence_max", 0.0)),
        segment_start_sec=float(best["segment_start_sec"]),
        segment_end_sec=float(best["segment_end_sec"]),
        source_scene_indices=tuple(int(x) for x in best.get("source_scene_indices", [])),
    )


def build_centered_window(
    selected: SelectedSegment,
    *,
    window_radius_sec: float,
    center_sec: float | None = None,
    video_duration_sec: float | None = None,
) -> Dict[str, float]:
    """
    Expand a segment into a centered review window.

    The new window is centered at the original segment midpoint and extends
    window_radius_sec to both sides. If video_duration_sec is provided, the
    window is clamped to [0, duration].
    """
    if window_radius_sec <= 0:
        raise ValueError("window_radius_sec must be positive")

    actual_center_sec = (
        float(center_sec)
        if center_sec is not None
        else (selected.segment_start_sec + selected.segment_end_sec) / 2.0
    )
    start_sec = actual_center_sec - window_radius_sec
    end_sec = actual_center_sec + window_radius_sec

    if video_duration_sec is not None:
        start_sec = max(0.0, start_sec)
        end_sec = min(video_duration_sec, end_sec)

    return {
        "center_sec": round(actual_center_sec, 3),
        "window_start_sec": round(start_sec, 3),
        "window_end_sec": round(end_sec, 3),
        "window_start_hhmmss": _seconds_to_hhmmss(start_sec),
        "window_end_hhmmss": _seconds_to_hhmmss(end_sec),
    }


def build_explicit_window(
    *,
    start_sec: float,
    end_sec: float,
    video_duration_sec: float | None = None,
) -> Dict[str, float]:
    """Build a manually specified review/export window."""
    if end_sec <= start_sec:
        raise ValueError("end_sec must be greater than start_sec")

    actual_start = float(start_sec)
    actual_end = float(end_sec)
    if video_duration_sec is not None:
        actual_start = max(0.0, actual_start)
        actual_end = min(video_duration_sec, actual_end)
        if actual_end <= actual_start:
            raise ValueError("Clamped explicit window is empty")

    return {
        "center_sec": round((actual_start + actual_end) / 2.0, 3),
        "window_start_sec": round(actual_start, 3),
        "window_end_sec": round(actual_end, 3),
        "window_start_hhmmss": _seconds_to_hhmmss(actual_start),
        "window_end_hhmmss": _seconds_to_hhmmss(actual_end),
    }


def _dense_sample_times(start_sec: float, end_sec: float, sample_every_sec: float) -> List[float]:
    if sample_every_sec <= 0:
        raise ValueError("sample_every_sec must be positive")
    if end_sec < start_sec:
        raise ValueError("end_sec must be >= start_sec")

    duration = end_sec - start_sec
    if duration == 0:
        return [round(start_sec, 3)]

    count = int(math.floor(duration / sample_every_sec)) + 1
    times = [round(start_sec + i * sample_every_sec, 3) for i in range(count)]
    if not math.isclose(times[-1], end_sec, abs_tol=1e-3):
        times.append(round(end_sec, 3))
    return times


def _video_metadata(video_path: str) -> Dict[str, float]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or frame_count <= 0:
            raise ValueError(f"Could not read positive fps/frame_count from: {video_path}")
        duration = frame_count / fps
        return {
            "fps": fps,
            "frame_count": frame_count,
            "duration_sec": duration,
        }
    finally:
        cap.release()


def _extract_frames_at_times(
    video_path: str,
    *,
    sample_times: Sequence[float],
    out_dir: str,
    image_quality: int = 95,
) -> List[Dict[str, Any]]:
    import cv2

    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        records: List[Dict[str, Any]] = []

        for sample_index, timestamp_sec in enumerate(sample_times):
            frame_index = int(round(timestamp_sec * fps))
            frame_index = max(0, min(frame_count - 1, frame_index))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise ValueError(
                    f"Failed to read frame at sample_index={sample_index} "
                    f"(timestamp={timestamp_sec}, frame_index={frame_index})"
                )

            filename = f"sample_{sample_index:04d}_t{timestamp_sec:08.3f}_f{frame_index:06d}.jpg"
            out_path = os.path.join(out_dir, filename)
            ok_write = cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), image_quality])
            if not ok_write:
                raise ValueError(f"Failed to write frame image: {out_path}")

            records.append(
                {
                    "sample_index": sample_index,
                    "timestamp_sec": round(timestamp_sec, 3),
                    "timestamp_hhmmss": _seconds_to_hhmmss(timestamp_sec),
                    "frame_index": frame_index,
                    "output_path": out_path,
                }
            )
        return records
    finally:
        cap.release()


def _export_continuous_clip(
    video_path: str,
    *,
    start_sec: float,
    end_sec: float,
    out_path: str,
) -> Dict[str, Any]:
    import cv2

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = max(0, min(frame_count - 1, int(round(start_sec * fps))))
        end_frame = max(start_frame, min(frame_count - 1, int(round(end_sec * fps))))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise ValueError(f"Could not open video writer: {out_path}")

        frames_written = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_index in range(start_frame, end_frame + 1):
            ok, frame = cap.read()
            if not ok or frame is None:
                raise ValueError(f"Failed to read frame {frame_index} while exporting clip")
            writer.write(frame)
            frames_written += 1
        writer.release()
        return {
            "output_path": out_path,
            "fps": round(fps, 3),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frames_written": frames_written,
        }
    finally:
        cap.release()


def _export_continuous_frames(
    video_path: str,
    *,
    start_sec: float,
    end_sec: float,
    out_dir: str,
    image_quality: int = 95,
) -> Dict[str, Any]:
    import cv2

    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = max(0, min(frame_count - 1, int(round(start_sec * fps))))
        end_frame = max(start_frame, min(frame_count - 1, int(round(end_sec * fps))))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        records: List[Dict[str, Any]] = []
        for frame_index in range(start_frame, end_frame + 1):
            ok, frame = cap.read()
            if not ok or frame is None:
                raise ValueError(f"Failed to read frame {frame_index} while exporting frame directory")
            timestamp_sec = frame_index / fps
            filename = f"frame_{frame_index:06d}_t{timestamp_sec:08.3f}.jpg"
            out_path = os.path.join(out_dir, filename)
            ok_write = cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), image_quality])
            if not ok_write:
                raise ValueError(f"Failed to write frame image: {out_path}")
            records.append(
                {
                    "frame_index": frame_index,
                    "timestamp_sec": round(timestamp_sec, 3),
                    "timestamp_hhmmss": _seconds_to_hhmmss(timestamp_sec),
                    "output_path": out_path,
                }
            )
        return {
            "fps": round(fps, 3),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frames_written": len(records),
            "frames": records,
        }
    finally:
        cap.release()


def _batched(items: Sequence[Dict[str, Any]], batch_size: int) -> Iterable[Sequence[Dict[str, Any]]]:
    for idx in range(0, len(items), batch_size):
        yield items[idx : idx + batch_size]


def _build_contact_sheets(
    frame_records: Sequence[Dict[str, Any]],
    *,
    out_dir: str,
    title_prefix: str,
    columns: int = 4,
    rows: int = 5,
    thumb_width: int = 320,
    thumb_height: int = 180,
    label_height: int = 28,
) -> List[str]:
    from PIL import Image, ImageDraw

    os.makedirs(out_dir, exist_ok=True)
    if columns <= 0 or rows <= 0:
        raise ValueError("columns and rows must be positive")
    page_size = columns * rows
    out_paths: List[str] = []

    for page_index, batch in enumerate(_batched(list(frame_records), page_size)):
        canvas = Image.new(
            "RGB",
            (columns * thumb_width, rows * (thumb_height + label_height)),
            color=(12, 12, 12),
        )
        draw = ImageDraw.Draw(canvas)
        for item_index, record in enumerate(batch):
            row = item_index // columns
            col = item_index % columns
            x = col * thumb_width
            y = row * (thumb_height + label_height)
            image = Image.open(record["output_path"]).convert("RGB")
            image = image.resize((thumb_width, thumb_height))
            canvas.paste(image, (x, y))
            label = f"{record['timestamp_hhmmss']} | f={record['frame_index']}"
            draw.text((x + 8, y + thumb_height + 6), label, fill=(235, 235, 235))

        out_path = os.path.join(out_dir, f"{title_prefix}_sheet_{page_index + 1:02d}.jpg")
        canvas.save(out_path, quality=92)
        out_paths.append(out_path)

    return out_paths


def deliberate_resample_object(
    segments_json: Dict[str, Any],
    *,
    object_id: str,
    video_path: str,
    out_dir: str,
    segment_id: str | None = None,
    window_radius_sec: float = 120.0,
    sample_every_sec: float = 1.0,
    center_sec: float | None = None,
) -> Dict[str, Any]:
    """
    Create a dense, centered review set for one object segment.

    Outputs:
        - selected segment metadata
        - centered review window
        - dense frame extraction records
        - contact sheet paths
    """
    meta = _video_metadata(video_path)
    selected = select_segment(segments_json, object_id=object_id, segment_id=segment_id)
    centered = build_centered_window(
        selected,
        window_radius_sec=window_radius_sec,
        center_sec=center_sec,
        video_duration_sec=meta["duration_sec"],
    )
    sample_times = _dense_sample_times(
        centered["window_start_sec"],
        centered["window_end_sec"],
        sample_every_sec=sample_every_sec,
    )

    object_slug = _slugify(selected.canonical_name)
    frames_dir = os.path.join(out_dir, "frames")
    contact_dir = os.path.join(out_dir, "contact_sheets")
    os.makedirs(out_dir, exist_ok=True)
    frame_records = _extract_frames_at_times(
        video_path,
        sample_times=sample_times,
        out_dir=frames_dir,
    )
    contact_sheet_paths = _build_contact_sheets(
        frame_records,
        out_dir=contact_dir,
        title_prefix=object_slug,
    )

    return {
        "meta": {
            "build_type": "deliberate_object_resampling",
            "video_path": video_path,
            "video_fps": round(meta["fps"], 3),
            "video_duration_sec": round(meta["duration_sec"], 3),
            "window_radius_sec": window_radius_sec,
            "sample_every_sec": sample_every_sec,
            "frames_returned": len(frame_records),
            "contact_sheets_returned": len(contact_sheet_paths),
        },
        "selected_segment": {
            "segment_id": selected.segment_id,
            "object_id": selected.object_id,
            "canonical_name": selected.canonical_name,
            "segment_role": selected.segment_role,
            "priority_score": round(selected.priority_score, 4),
            "alignment_confidence_max": round(selected.alignment_confidence_max, 4),
            "segment_start_sec": round(selected.segment_start_sec, 3),
            "segment_end_sec": round(selected.segment_end_sec, 3),
            "segment_start_hhmmss": _seconds_to_hhmmss(selected.segment_start_sec),
            "segment_end_hhmmss": _seconds_to_hhmmss(selected.segment_end_sec),
            "source_scene_indices": list(selected.source_scene_indices),
        },
        "centered_window": centered,
        "frames": frame_records,
        "contact_sheets": contact_sheet_paths,
    }


def refine_explicit_window(
    *,
    label: str,
    video_path: str,
    out_dir: str,
    start_sec: float,
    end_sec: float,
    review_sample_every_sec: float = 0.5,
) -> Dict[str, Any]:
    """
    Export a manual refinement package for one explicit short clip.

    Outputs:
        - sparse review frames at review_sample_every_sec
        - review contact sheets
        - continuous mp4 clip
        - continuous frame directory
    """
    meta = _video_metadata(video_path)
    window = build_explicit_window(
        start_sec=start_sec,
        end_sec=end_sec,
        video_duration_sec=meta["duration_sec"],
    )

    sample_times = _dense_sample_times(
        window["window_start_sec"],
        window["window_end_sec"],
        sample_every_sec=review_sample_every_sec,
    )

    label_slug = _slugify(label)
    review_frames_dir = os.path.join(out_dir, "frames")
    contact_dir = os.path.join(out_dir, "contact_sheets")
    clip_dir = os.path.join(out_dir, "clip")
    continuous_frames_dir = os.path.join(out_dir, "continuous_frames")
    clip_path = os.path.join(clip_dir, f"{label_slug}.mp4")

    os.makedirs(out_dir, exist_ok=True)
    review_frame_records = _extract_frames_at_times(
        video_path,
        sample_times=sample_times,
        out_dir=review_frames_dir,
    )
    contact_sheet_paths = _build_contact_sheets(
        review_frame_records,
        out_dir=contact_dir,
        title_prefix=label_slug,
    )
    clip_info = _export_continuous_clip(
        video_path,
        start_sec=window["window_start_sec"],
        end_sec=window["window_end_sec"],
        out_path=clip_path,
    )
    continuous_frame_info = _export_continuous_frames(
        video_path,
        start_sec=window["window_start_sec"],
        end_sec=window["window_end_sec"],
        out_dir=continuous_frames_dir,
    )

    return {
        "meta": {
            "build_type": "manual_explicit_window_refinement",
            "label": label,
            "video_path": video_path,
            "video_fps": round(meta["fps"], 3),
            "video_duration_sec": round(meta["duration_sec"], 3),
            "review_sample_every_sec": review_sample_every_sec,
            "review_frames_returned": len(review_frame_records),
            "contact_sheets_returned": len(contact_sheet_paths),
        },
        "window": window,
        "review_frames": review_frame_records,
        "contact_sheets": contact_sheet_paths,
        "continuous_clip": clip_info,
        "continuous_frames": continuous_frame_info,
    }
