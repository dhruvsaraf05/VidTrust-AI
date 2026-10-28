"""The link path: sampling coverage, SSRF refusal, and cookie scoping."""
from __future__ import annotations

import pytest

import config
from pipeline import url_fetcher
from pipeline.video_sampler import sample_frames

from conftest import write_clip


def test_short_clip_sampling_is_one_frame_per_second(tmp_path):
    frames, info = sample_frames(str(write_clip(tmp_path / "c.mp4", 13, 10)))
    assert len(frames) == 13
    assert info["subsampled"] is False


def test_long_clip_is_spread_across_its_whole_duration(tmp_path):
    seconds = 2 * config.VIDEO_MAX_FRAMES
    frames, info = sample_frames(str(write_clip(tmp_path / "c.mp4", seconds, 2)))
    assert len(frames) == config.VIDEO_MAX_FRAMES
    assert info["subsampled"] is True
    # The last sample lands in the final stretch, not at the 60 s mark.
    assert frames[-1]["timestamp"] >= seconds - 3


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:8000/",
    "http://localhost/",
    "http://169.254.169.254/latest/meta-data/",
    "http://10.0.0.5/",
    "http://[::1]/",
])
def test_non_public_hosts_are_refused(url):
    allowed, _ = url_fetcher._resolves_to_public_host(url)
    assert allowed is False


def test_refusal_does_not_reach_ytdlp(monkeypatch):
    import yt_dlp

    def forbidden(*args, **kwargs):
        raise AssertionError("yt-dlp was handed a private address")
    monkeypatch.setattr(yt_dlp, "YoutubeDL", forbidden)
    with pytest.raises(url_fetcher.UrlError) as caught:
        url_fetcher.fetch("http://127.0.0.1:8000/api/health", "unused")
    assert caught.value.code == "URL_INVALID"


@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=x",
    "https://evil.example/instagram.com/reel/x",
    "https://instagram.com.evil.example/reel/x",
])
def test_cookies_never_leave_the_login_walled_hosts(url):
    assert url_fetcher._cookie_options(url) == {"cookiefile": None,
                                                "cookiesfrombrowser": None}


def test_health_reports_the_real_url_limits(client):
    url = client.get("/api/health").json()["url"]
    assert url["max_duration_s"] == config.URL_MAX_DURATION_S
    assert url["max_height"] == config.URL_MAX_HEIGHT
