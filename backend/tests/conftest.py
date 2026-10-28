"""Shared fixtures. Nothing here loads the classifier.

The suite runs without the 744 MB checkpoint: the TestClient is created WITHOUT
its context manager, so the lifespan handler never fires and the model stays
unloaded. That is not a shortcut -- it is invariant 6 under test. Where a test
needs classifier scores, it stubs model_detector at the boundary.

    cd backend && .venv/Scripts/python -m pytest
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    import main
    # No `with`: the lifespan (and so the model load) is deliberately skipped.
    return TestClient(main.app, raise_server_exceptions=False)


@pytest.fixture
def png_bytes(tmp_path) -> bytes:
    """A small synthetic PNG. Plumbing only -- says nothing about accuracy."""
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    path = tmp_path / "plumbing.png"
    Image.fromarray(pixels).save(path)
    return path.read_bytes()


def write_clip(path: Path, seconds: int, fps: int, size: int = 32) -> Path:
    """Write a tiny synthetic clip, or skip if this OpenCV build cannot encode."""
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (size, size))
    if not writer.isOpened():
        pytest.skip("OpenCV cannot write mp4v on this machine")
    for i in range(seconds * fps):
        frame = np.full((size, size, 3), i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path
