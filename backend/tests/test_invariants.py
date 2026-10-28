"""CLAUDE.md section 6, as executable checks. One test (or group) per invariant.

These encode decisions made deliberately, several after finding a bug. A
failure here means a design rule was broken, not that a number moved.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import config
from pipeline import aggregator
from detectors import model_detector

from conftest import BACKEND, write_clip

FROZEN_KEYS = {"id", "filename", "media_type", "verdict", "confidence",
               "processing_time_ms", "signals", "frames"}
SIGNAL_KEYS = {"model", "metadata", "frequency"}
SIGNAL_FIELDS = {"name", "score", "weight", "detail", "available"}


def _signal(score, weight, available=True):
    return {"score": score, "weight": weight, "available": available}


# 1 -------------------------------------------------------------------------
def test_unavailable_signal_is_renormalised_out_not_scored_zero():
    signals = {
        "model": _signal(0.9, 0.60),
        "metadata": _signal(0.0, 0.25, available=False),
        "frequency": _signal(0.5, 0.15),
    }
    expected = (0.9 * 0.60 + 0.5 * 0.15) / (0.60 + 0.15)
    assert aggregator.fuse(signals) == pytest.approx(expected)
    # Scoring it as 0.0 would have given 0.615 -- UNCERTAIN instead of AI.
    assert aggregator.verdict_for(aggregator.fuse(signals)) == "AI_GENERATED"


def test_unavailable_score_is_ignored_whatever_it_holds():
    base = {"model": _signal(0.8, 0.60), "frequency": _signal(0.4, 0.15)}
    low = aggregator.fuse({**base, "metadata": _signal(0.0, 0.25, False)})
    high = aggregator.fuse({**base, "metadata": _signal(1.0, 0.25, False)})
    assert low == high


# 2 -------------------------------------------------------------------------
@pytest.mark.parametrize("labels", [
    ["LABEL_0", "LABEL_1"],
    ["class_a", "class_b"],
])
def test_generic_labels_are_never_mapped(labels):
    predictions = [{"label": label, "score": 0.5} for label in labels]
    assert model_detector._ai_probability(predictions) is None


def test_known_labels_map_in_the_right_direction():
    assert model_detector._ai_probability(
        [{"label": "artificial", "score": 0.9}, {"label": "human", "score": 0.1}]
    ) == pytest.approx(0.9)
    # A head that only names the real class is inverted, not taken at face value.
    assert model_detector._ai_probability(
        [{"label": "human", "score": 0.8}]
    ) == pytest.approx(0.2)


def test_unrecognised_vocabulary_sets_available_false(monkeypatch):
    monkeypatch.setattr(model_detector, "_pipe", object())
    monkeypatch.setattr(model_detector, "score_images", lambda images: [None])
    result = model_detector.analyze(object())
    assert result["available"] is False


# 3 -------------------------------------------------------------------------
def test_classifier_loads_only_in_the_lifespan_handler():
    calls = []
    for path in [BACKEND / "main.py", *BACKEND.glob("pipeline/*.py"),
                 *BACKEND.glob("detectors/*.py")]:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"\bload_model\(\)", line) and "def load_model" not in line:
                calls.append(f"{path.name}:{number}")
    assert len(calls) == 1 and calls[0].startswith("main.py"), calls
    source = (BACKEND / "main.py").read_text(encoding="utf-8")
    lifespan = source[source.index("async def lifespan"):source.index("app = FastAPI(")]
    assert "load_model()" in lifespan


def test_request_path_never_triggers_a_load(monkeypatch, client, png_bytes):
    def refuse():
        raise AssertionError("load_model called on the request path")
    monkeypatch.setattr(model_detector, "load_model", refuse)
    response = client.post("/api/analyze", files={"file": ("a.png", png_bytes)})
    assert response.status_code == 200


# 4 -------------------------------------------------------------------------
def _assert_contract_error(response, code):
    body = response.json()
    assert set(body) == {"error", "message"}, body
    assert body["error"] == code
    assert "detail" not in body


def test_missing_file_field_stays_in_contract_shape(client):
    response = client.post("/api/analyze", data={"wrong": "x"})
    assert response.status_code == 415
    _assert_contract_error(response, "UNSUPPORTED_FORMAT")


def test_bad_extension_and_empty_file(client):
    _assert_contract_error(
        client.post("/api/analyze", files={"file": ("a.exe", b"MZ")}),
        "UNSUPPORTED_FORMAT")
    _assert_contract_error(
        client.post("/api/analyze", files={"file": ("a.png", b"")}),
        "UNSUPPORTED_FORMAT")


def test_url_validation_error_stays_in_contract_shape(client):
    response = client.post("/api/analyze-url", json={"nope": 1})
    _assert_contract_error(response, "URL_INVALID")


def test_corrupt_media_is_processing_failed(client):
    response = client.post("/api/analyze", files={"file": ("a.png", b"not a png")})
    assert response.status_code == 500
    _assert_contract_error(response, "PROCESSING_FAILED")


# 5 -------------------------------------------------------------------------
SECRET_PATH = r"C:\Users\server\secret\upload_tmp123.png"


def test_analysis_failure_does_not_leak_paths(monkeypatch, client, png_bytes):
    def explode(path):
        raise FileNotFoundError(f"[Errno 2] No such file: '{SECRET_PATH}'")
    monkeypatch.setattr(aggregator, "analyze_image", explode)
    response = client.post("/api/analyze", files={"file": ("a.png", png_bytes)})
    _assert_contract_error(response, "PROCESSING_FAILED")
    assert "secret" not in response.json()["message"]


def test_unhandled_exception_does_not_leak_paths(monkeypatch, client):
    import main

    def explode(*args, **kwargs):
        raise OSError(f"cannot open '{SECRET_PATH}'")
    monkeypatch.setattr(main, "fetch_url", explode)
    response = client.post("/api/analyze-url", json={"url": "https://example.com/v"})
    body = response.json()
    assert body["error"] == "PROCESSING_FAILED"
    assert "secret" not in body["message"] and "Users" not in body["message"]


# 6 -------------------------------------------------------------------------
def test_dead_classifier_is_not_a_500(client, png_bytes):
    assert not model_detector.is_available()
    response = client.post("/api/analyze", files={"file": ("a.png", png_bytes)})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == FROZEN_KEYS
    assert set(body["signals"]) == SIGNAL_KEYS
    for signal in body["signals"].values():
        assert set(signal) == SIGNAL_FIELDS
    assert body["signals"]["model"]["available"] is False
    assert body["frames"] is None


def test_model_unavailable_only_on_health(client):
    health = client.get("/api/health").json()
    assert health["error"] == "MODEL_UNAVAILABLE"
    assert health["status"] == "degraded"
    source = (BACKEND / "main.py").read_text(encoding="utf-8")
    assert 'api_error("MODEL_UNAVAILABLE"' not in source


# 7 -------------------------------------------------------------------------
def test_no_faces_detected_is_unreachable():
    hits = []
    for path in BACKEND.rglob("*.py"):
        if ".venv" in path.parts or "tests" in path.parts:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "NO_FACES_DETECTED" in line and not line.lstrip().startswith("#"):
                hits.append(f"{path.name}:{number}: {line.strip()}")
    # Exactly one non-comment mention: the contract table entry.
    assert len(hits) == 1 and '"NO_FACES_DETECTED": 422' in hits[0], hits


# 8 -------------------------------------------------------------------------
def test_frame_scores_are_classifier_only(monkeypatch, tmp_path):
    clip = write_clip(tmp_path / "clip.mp4", seconds=5, fps=10)
    monkeypatch.setattr(model_detector, "is_available", lambda: True)
    monkeypatch.setattr(model_detector, "model_name", lambda: "stub")
    monkeypatch.setattr(model_detector, "score_images",
                        lambda images: [0.11 * (i + 1) for i in range(len(images))])
    result = aggregator.analyze_video(str(clip))
    scores = [frame["score"] for frame in result["frames"]]
    assert scores == pytest.approx([0.11 * (i + 1) for i in range(len(scores))])
    # Fused confidence differs from every per-frame score: they are not fused.
    assert result["signals"]["model"]["available"] is True


# 9 -------------------------------------------------------------------------
def test_all_signals_unavailable_is_uncertain_not_real():
    signals = {key: _signal(0.0, config.WEIGHTS[key], available=False)
               for key in SIGNAL_KEYS}
    assert aggregator.fuse(signals) == 0.5
    assert aggregator.verdict_for(0.5) == "UNCERTAIN"


# 10 ------------------------------------------------------------------------
def test_evaluation_set_and_samples_are_disjoint():
    import hashlib

    def digests(folder: Path) -> set[str]:
        return {hashlib.sha256(p.read_bytes()).hexdigest()
                for p in folder.rglob("*")
                if p.is_file() and p.suffix.lower() in config.ALLOWED_EXTENSIONS
                and not any(part.startswith("_") for part in p.relative_to(folder).parts)}

    images = BACKEND / "evaluation_set" / "images"
    if not images.exists():
        pytest.skip("evaluation set not built on this machine (gitignored)")
    overlap = digests(images) & digests(BACKEND / "samples")
    assert not overlap


# /api/explain is additive and obeys the same rules ------------------------
def test_explain_with_dead_classifier_declares_unavailable(client, png_bytes):
    response = client.post("/api/explain", files={"file": ("a.png", png_bytes)})
    assert response.status_code == 200
    saliency = response.json()["saliency"]
    assert saliency["available"] is False
    assert saliency["heatmap_png_b64"] is None


def test_explain_refuses_video_in_contract_shape(client):
    response = client.post("/api/explain", files={"file": ("a.mp4", b"\x00" * 64)})
    assert response.status_code == 415
    _assert_contract_error(response, "UNSUPPORTED_FORMAT")


def test_explain_corrupt_image_is_processing_failed(client):
    response = client.post("/api/explain", files={"file": ("a.png", b"not a png")})
    _assert_contract_error(response, "PROCESSING_FAILED")
