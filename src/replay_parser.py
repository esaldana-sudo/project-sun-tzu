#!/usr/bin/env python3
"""
replay_parser.py
----------------
Basic video frame extractor that samples frames every N milliseconds and (by default)
writes an index CSV alongside the images.

Usage examples:
    # Extract a frame every 250 ms into data/processed/frames/<replay_stem>/
    python -m src.replay_parser "data/replays/2025-08-16 12-33-04.mp4" --every-ms 250

    # Extract a 10s clip starting at 30s, resize to width 1280, and skip CSV
    python -m src.replay_parser "data/replays/foo.mp4" --every-ms 200 --start-ms 30000 \
        --duration-ms 10000 --resize-width 1280 --no-index
"""

from __future__ import annotations
import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Optional

import cv2
from tqdm import tqdm


# -----------------------
# Path / project utilities
# -----------------------

def project_root() -> Path:
    """Return the repository root by walking up until we find markers (src/ & data/)."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "src").is_dir() and (parent / "data").is_dir():
            return parent
    # Fallback: two levels up from this file
    return Path(__file__).resolve().parents[1]


def relpath(p: Path) -> Path:
    """Best-effort pretty-print of a path relative to repo root for logs."""
    try:
        return p.relative_to(project_root())
    except ValueError:
        return p


# -----------------------
# Arg parsing helpers
# -----------------------

def positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return ivalue


def nonneg_int(value: str) -> int:
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("Must be a non-negative integer.")
    return ivalue


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract frames from a replay video at a fixed interval (ms). "
            "Paths are resolved relative to the repository root."
        )
    )
    parser.add_argument(
        "video",
        type=str,
        help="Path to input video (e.g., data/replays/your_file.mp4).",
    )
    parser.add_argument(
        "--every-ms",
        type=positive_int,
        default=250,
        help="Save one frame every N milliseconds. Default: 250",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/processed/frames",
        help=(
            "Base output directory. A subfolder named after the video stem will be created. "
            "Default: data/processed/frames"
        ),
    )
    parser.add_argument(
        "--start-ms",
        type=nonneg_int,
        default=0,
        help="Start extracting at this timestamp (ms). Default: 0",
    )
    parser.add_argument(
        "--duration-ms",
        type=nonneg_int,
        default=0,
        help="If > 0, only extract for this many ms (from start). Default: 0 (to end)",
    )
    parser.add_argument(
        "--resize-width",
        type=positive_int,
        default=0,
        help="Optional: resize output frames to this width (keeps aspect ratio). Default: 0 (no resize)",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        help="JPEG quality 0-100. Default: 95",
    )
    # Boolean flag to *disable* index writing; default behavior is to write CSV
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="If set, do NOT write frames.csv alongside images.",
    )
    return parser.parse_args(argv)


# -----------------------
# Core helpers
# -----------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def ms_to_frame_index(ms: int, fps: float) -> int:
    """Convert a millisecond timestamp to a frame index using floor."""
    return max(0, int(math.floor((ms / 1000.0) * fps)))


def read_video_meta(cap: cv2.VideoCapture) -> tuple[float, int]:
    """Return (fps, frame_count)."""
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return fps, frame_count


def maybe_resize(frame, resize_width: int):
    if resize_width and resize_width > 0:
        h, w = frame.shape[:2]
        if w != resize_width:
            scale = resize_width / float(w)
            new_h = max(1, int(round(h * scale)))
            frame = cv2.resize(frame, (resize_width, new_h), interpolation=cv2.INTER_AREA)
    return frame


def write_frame(out_dir: Path, seq_num: int, frame, imwrite_params) -> int:
    out_path = out_dir / f"frame_{seq_num:06d}.jpg"
    ok = cv2.imwrite(str(out_path), frame, imwrite_params)
    return 1 if ok else 0


def write_index_csv(index_rows: list[dict], out_dir: Path) -> None:
    csv_path = out_dir / "frames.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["seq_num", "frame_idx", "ms", "filename"]
        )
        writer.writeheader()
        writer.writerows(index_rows)


# -----------------------
# Extraction
# -----------------------

def extract_frames(
    video_path: Path,
    out_base_dir: Path,
    every_ms: int,
    start_ms: int,
    duration_ms: int,
    resize_width: int,
    jpg_quality: int,
    write_index: bool = True,
) -> int:
    """Core extraction loop. Returns number of frames written."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps, frame_count = read_video_meta(cap)
    if fps <= 1e-6:
        cap.release()
        raise RuntimeError(
            "FPS is invalid (0). Try re-encoding the video or check the file."
        )

    # Determine time range (ms)
    video_duration_ms = int((frame_count / fps) * 1000) if frame_count > 0 else 0
    if duration_ms > 0:
        end_ms = start_ms + duration_ms
    elif video_duration_ms > 0:
        end_ms = video_duration_ms
    else:
        end_ms = None  # unknown end; iterate until reads fail

    # Prepare output dir
    out_dir = out_base_dir / video_path.stem
    ensure_dir(out_dir)

    # Prepare image write params
    jpg_quality = max(0, min(jpg_quality, 100))
    imwrite_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality]

    # Precompute intended frame indices if we know the end
    intended_indices = None
    if end_ms is not None:
        intended_indices = []
        t = start_ms
        while t <= end_ms:
            intended_indices.append(ms_to_frame_index(t, fps))
            t += every_ms

    saved = 0
    last_saved_idx = -1
    index_rows: list[dict] = []

    if intended_indices is None:
        # Unknown end: advance timestamps until reads fail
        pbar = tqdm(desc=f"Extracting (unknown length) → {relpath(out_dir)}", unit="frame")
        t = start_ms
        while True:
            idx = ms_to_frame_index(t, fps)
            if idx == last_saved_idx:
                t += every_ms
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            frame = maybe_resize(frame, resize_width)
            saved += write_frame(out_dir, saved + 1, frame, imwrite_params)
            index_rows.append({
                "seq_num": saved,
                "frame_idx": idx,
                "ms": t,
                "filename": f"frame_{saved:06d}.jpg",
            })
            last_saved_idx = idx
            pbar.update(1)
            t += every_ms
        pbar.close()
    else:
        # Known end: iterate over precomputed indices
        pbar = tqdm(total=len(intended_indices), desc=f"Extracting → {relpath(out_dir)}", unit="frame")
        for idx in intended_indices:
            if idx == last_saved_idx:
                pbar.update(1)
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                pbar.update(1)
                continue
            frame = maybe_resize(frame, resize_width)
            saved += write_frame(out_dir, saved + 1, frame, imwrite_params)
            index_rows.append({
                "seq_num": saved,
                "frame_idx": idx,
                # Align ms to actual frame index for consistency
                "ms": int(round((idx / fps) * 1000)),
                "filename": f"frame_{saved:06d}.jpg",
            })
            last_saved_idx = idx
            pbar.update(1)
        pbar.close()

    cap.release()

    if write_index:
        write_index_csv(index_rows, out_dir)

    return saved


# -----------------------
# Main entry
# -----------------------

def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    root = project_root()
    video_path = (root / args.video).resolve()
    out_base_dir = (root / args.out_dir).resolve()

    if not video_path.exists():
        print(f"[ERROR] Video not found: {relpath(video_path)}", file=sys.stderr)
        return 1

    ensure_dir(out_base_dir)

    try:
        saved = extract_frames(
            video_path=video_path,
            out_base_dir=out_base_dir,
            every_ms=args.every_ms,
            start_ms=args.start_ms,
            duration_ms=args.duration_ms,
            resize_width=args.resize_width,
            jpg_quality=args.jpg_quality,
            write_index=not args.no_index,
        )
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    print(f"[OK] Saved {saved} frames to {relpath(out_base_dir / video_path.stem)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
