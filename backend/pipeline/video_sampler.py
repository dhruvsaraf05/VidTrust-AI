"""Frame sampling for video input.

Video is treated as images over time: pull roughly one frame per second, hard
capped, and hand them to the same detectors the image path uses.
"""

from __future__ import annotations

import logging

import cv2
from PIL import Image

import config

logger = logging.getLogger(__name__)


class VideoReadError(RuntimeError):
    """Raised when the container cannot be opened or yields no frames."""


def sample_frames(path: str) -> tuple[list[dict], dict]:
    """Sample frames at ~VIDEO_SAMPLE_FPS, capped at VIDEO_MAX_FRAMES.

    Returns (frames, info) where each frame is
    {"index": int, "timestamp": float, "image": PIL.Image} and info carries
    duration/fps/coverage for the detail strings.

    TODO(roadmap): when a clip is longer than VIDEO_MAX_FRAMES seconds we
    currently analyse only its first 60 seconds. Spreading the sample evenly
    across the full duration would be a better use of the same 60-frame budget.
    """
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise VideoReadError("Could not open video container")

    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        total_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)

        # Some containers report nonsense here; fall back to a sane default
        # rather than dividing by zero.
        if not fps or fps <= 0 or fps != fps:  # NaN-safe
            logger.warning("Video reports fps=%r; assuming 25.0", fps)
            fps = 25.0

        duration = (total_frames / fps) if total_frames and total_frames > 0 else 0.0
        step = max(1, int(round(fps / config.VIDEO_SAMPLE_FPS)))

        frames: list[dict] = []
        position = 0
        truncated = False

        while True:
            ok = capture.grab()
            if not ok:
                break

            if position % step == 0:
                ok, raw = capture.retrieve()
                if ok and raw is not None:
                    rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                    frames.append({
                        "index": len(frames),
                        "timestamp": round(position / fps, 2),
                        "image": Image.fromarray(rgb),
                    })
                    if len(frames) >= config.VIDEO_MAX_FRAMES:
                        truncated = True
                        break

            position += 1
    finally:
        capture.release()

    if not frames:
        raise VideoReadError("Video contained no readable frames")

    covered = frames[-1]["timestamp"]
    info = {
        "fps": round(float(fps), 2),
        "duration_s": round(float(duration), 2),
        "sampled": len(frames),
        "truncated": truncated,
        "covered_s": covered,
    }
    logger.info("Sampled %d frames from %s (%s)", len(frames), path, info)
    return frames, info
