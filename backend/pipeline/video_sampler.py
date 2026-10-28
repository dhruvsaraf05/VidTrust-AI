"""Frame sampling for video input.

Video is treated as images over time: pull roughly one frame per second, hard
capped, and hand them to the same detectors the image path uses.
"""

from __future__ import annotations

import logging
import math

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

        # Stride wide enough that VIDEO_MAX_FRAMES samples SPAN THE WHOLE CLIP,
        # rather than taking the first 60 s and stopping.
        #
        # The old rule was the 1 fps stride alone, so a 10-minute video was
        # judged entirely on its opening minute and the verdict was silently
        # partial. Widening the stride costs nothing: the expensive part is the
        # classifier, which runs once per SAMPLED frame, so coverage improves
        # while inference cost stays flat.
        #
        # Short clips are unaffected -- a 13 s clip at 30 fps still gives
        # max(30, ceil(390/60)=7) = 30, i.e. the same 13 frames as before.
        #
        # Containers that lie about their length (total_frames <= 0, handled
        # above for fps) fall back to the fps-only stride and the hard cap.
        fps_step = fps / config.VIDEO_SAMPLE_FPS
        if total_frames and total_frames > 0:
            span_step = math.ceil(total_frames / config.VIDEO_MAX_FRAMES)
            step = max(1, int(round(max(fps_step, span_step))))
        else:
            step = max(1, int(round(fps_step)))

        frames: list[dict] = []
        position = 0
        # True when the stride was widened past 1 fps to cover the whole clip,
        # i.e. the sample is sparser than 1 fps. NOT "we stopped early" -- that
        # is what it used to mean, and the distinction matters because the
        # detail string shown to the user asserts one or the other.
        subsampled = False

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
                        break

            position += 1
    finally:
        capture.release()

    if not frames:
        raise VideoReadError("Video contained no readable frames")

    covered = frames[-1]["timestamp"]
    subsampled = step > int(round(fps_step))
    info = {
        "fps": round(float(fps), 2),
        "duration_s": round(float(duration), 2),
        "sampled": len(frames),
        "subsampled": subsampled,
        "sample_interval_s": round(step / fps, 2),
        "covered_s": covered,
    }
    logger.info("Sampled %d frames from %s (%s)", len(frames), path, info)
    return frames, info
