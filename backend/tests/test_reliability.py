"""The Reliability Card reads the input's condition and quotes the study."""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

import reliability


def _jpeg(quality: int) -> Image.Image:
    rng = np.random.default_rng(1)
    buffer = io.BytesIO()
    Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(
        buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer)


@pytest.mark.parametrize("quality", [95, 85, 75, 60, 50, 30])
def test_jpeg_quality_estimate_recovers_libjpeg_quality(quality):
    assert abs(reliability.estimate_jpeg_quality(_jpeg(quality)) - quality) <= 2


def test_very_low_quality_still_maps_to_the_lowest_measured_condition():
    # Table entries clip at 255 below ~q15, so the estimate reads high there.
    estimate = reliability.estimate_jpeg_quality(_jpeg(10))
    assert estimate < 30
    observed = {"via": "upload", "media_type": "image", "jpeg_quality": estimate,
                "format": "JPEG", "platform": None}
    assert reliability.nearest_condition(observed)[0] == "jpeg30"


def test_png_has_no_jpeg_quality(tmp_path):
    path = tmp_path / "a.png"
    Image.new("RGB", (8, 8)).save(path)
    with Image.open(path) as image:
        assert reliability.estimate_jpeg_quality(image) is None


@pytest.mark.parametrize("quality, condition", [
    (97, "jpeg95"), (80, "jpeg75"), (55, "jpeg50"), (20, "jpeg30")])
def test_nearest_condition_for_jpeg(quality, condition):
    observed = {"via": "upload", "media_type": "image", "jpeg_quality": quality,
                "format": "JPEG", "platform": None}
    assert reliability.nearest_condition(observed)[0] == condition


def test_url_and_video_map_to_the_platform_condition():
    for observed in ({"via": "url", "media_type": "video", "platform": "Instagram"},
                     {"via": "upload", "media_type": "video", "platform": None}):
        assert reliability.nearest_condition(observed)[0] == "messaging"


def test_card_quotes_the_study_and_its_caveats(tmp_path):
    if reliability._study() is None:
        pytest.skip("robustness.json not generated on this machine")
    path = tmp_path / "a.jpg"
    _jpeg(75).save(path, format="JPEG", quality=75)
    result = reliability.card(str(path), "image")
    assert result["available"] is True
    assert result["condition"] == "jpeg75"
    study = reliability._study()["jpeg75"]["at_frozen_band"]["confusion"]
    assert result["measured"]["generated_flagged"] == study["tp"]
    assert any("upper bound" in caveat for caveat in result["caveats"])


def test_video_card_says_video_was_not_measured(tmp_path):
    if reliability._study() is None:
        pytest.skip("robustness.json not generated on this machine")
    result = reliability.card(str(tmp_path / "unused.mp4"), "video")
    assert "not covered" in result["caveats"][0]


def test_reliability_endpoint_is_additive_and_model_free(client, tmp_path):
    if reliability._study() is None:
        pytest.skip("robustness.json not generated on this machine")
    path = tmp_path / "a.jpg"
    _jpeg(50).save(path, format="JPEG", quality=50)
    response = client.post("/api/reliability",
                           files={"file": ("a.jpg", path.read_bytes())})
    assert response.status_code == 200
    assert response.json()["reliability"]["condition"] == "jpeg50"


def test_reliability_corrupt_file_is_declared_not_raised(client):
    response = client.post("/api/reliability", files={"file": ("a.jpg", b"junk")})
    assert response.status_code == 200
    assert response.json()["reliability"]["available"] is False
    assert "Users" not in response.json()["reliability"]["detail"]
