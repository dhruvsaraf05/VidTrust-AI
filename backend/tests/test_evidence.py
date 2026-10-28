"""The evidence drill-downs show what the signals actually measured."""
from __future__ import annotations

import json

import pytest
from PIL import Image

import config
from detectors import frequency_detector, metadata_detector

from conftest import BACKEND

SAMPLES = BACKEND / "samples"


def _committed_metadata():
    report = json.loads((BACKEND / "samples_report.json").read_text(encoding="utf-8"))
    return {row["filename"]: row["signals"]["metadata"]
            for row in report["results"] if row.get("ok")}


@pytest.mark.parametrize("name, committed", sorted(_committed_metadata().items()))
def test_metadata_refactor_reproduces_the_committed_sweep(name, committed):
    """The field collector was restructured for the drill-down. Every sample's
    provenance reading must still match the committed samples_report.json."""
    path = SAMPLES / name
    media = "video" if path.suffix.lower() in config.VIDEO_EXTENSIONS else "image"
    result = metadata_detector.analyze(str(path), media)
    assert (result["available"], result["score"], result["detail"]) == (
        committed["available"], committed["score"], committed["detail"])


def test_evidence_signal_is_the_analyze_result():
    path = str(SAMPLES / "_smoke_tagged.jpg")
    evidence = metadata_detector.evidence(path, "image")
    assert evidence["signal"] == metadata_detector.analyze(path, "image")
    assert any(field["matched"] == "midjourney" for field in evidence["fields"])


def test_binary_blocks_are_summarised_not_dumped():
    evidence = metadata_detector.evidence(str(SAMPLES / "IMG_5019.jpg"), "image")
    block = next(f for f in evidence["fields"] if f["source"] == "EXIF block")
    assert block["value"].startswith("binary block")
    assert {"key": "Make", "value": "Apple"}.items() <= next(
        f for f in evidence["fields"] if f["key"] == "Make").items()


def test_spectrum_score_is_the_signal_score():
    with Image.open(SAMPLES / "IMG_5019.jpg") as image:
        rgb = image.convert("RGB")
    picture = frequency_detector.spectrum(rgb)
    assert picture["available"] is True
    assert picture["score"] == frequency_detector.analyze(rgb)["score"]


def test_explain_returns_all_three_even_with_a_dead_classifier(client):
    payload = (SAMPLES / "_smoke_tagged.jpg").read_bytes()
    body = client.post("/api/explain", files={"file": ("a.jpg", payload)}).json()
    assert body["saliency"]["available"] is False
    assert body["spectrum"]["available"] is True
    assert body["provenance"]["signal"]["available"] is True


def test_heatmap_is_not_drawn_below_the_faithfulness_floor(monkeypatch):
    """The gate fires before any model call, so this needs no classifier."""
    from detectors import saliency

    monkeypatch.setattr(saliency, "faithfulness",
                        lambda: {"n": 20, "rho_median": -0.02})
    monkeypatch.setattr(saliency.model_detector, "pipe",
                        lambda: (_ for _ in ()).throw(AssertionError("model touched")))
    result = saliency.explain(Image.new("RGB", (8, 8)))
    assert result["available"] is False
    assert "-0.02" in result["detail"]


def test_heatmap_is_not_drawn_when_unmeasured(monkeypatch):
    from detectors import saliency

    monkeypatch.setattr(saliency, "faithfulness", lambda: None)
    result = saliency.explain(Image.new("RGB", (8, 8)))
    assert result["available"] is False
    assert "not been measured" in result["detail"]
