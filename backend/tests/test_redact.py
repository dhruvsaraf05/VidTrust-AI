"""Invariant 5 at the string level: paths go, everything else stays."""
from __future__ import annotations

import pytest

from detectors.redact import redact


@pytest.mark.parametrize("text, expected", [
    (r"No such file: 'C:\Users\HP\.cache\huggingface\hub\x.bin'", "No such file: '<path>'"),
    ("cannot open D:/data/upload.png now", "cannot open <path> now"),
    (r"denied: \\server\share\f.png", "denied: <path>"),
    ("cannot open /home/u/.cache/hf/model.safetensors now", "cannot open <path> now"),
    ("couldn't connect to https://huggingface.co/api/models",
     "couldn't connect to https://huggingface.co/api/models"),
    ("shape (1, 3) vs 2/3", "shape (1, 3) vs 2/3"),
    ("ratio 0.5 / 1.0", "ratio 0.5 / 1.0"),
])
def test_redact(text, expected):
    assert redact(text) == expected
