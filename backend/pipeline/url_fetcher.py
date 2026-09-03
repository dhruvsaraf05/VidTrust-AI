"""Fetch media from a URL with yt-dlp, for POST /api/analyze-url.

This module only *acquires* media. It contains no analysis: once a file is on
disk it is handed to the same video pipeline an upload goes through, so a URL
and an upload of the same clip produce the same verdict by construction.

Limits are enforced BEFORE or DURING the download, never after:

  * duration   -- read from the metadata probe and rejected before any media
                  bytes are fetched
  * resolution -- capped by the format selector, so 720p is what gets
                  requested rather than what gets discarded
  * size       -- yt-dlp's max_filesize aborts mid-download
  * wall clock -- a progress hook raises once the deadline passes

No ffmpeg
---------
ffmpeg is not assumed to be present. yt-dlp merges separate video and audio
streams with ffmpeg, so the format selector deliberately asks for a single
pre-muxed file. Audio is irrelevant here anyway -- the pipeline samples frames.

No authentication of any kind is attempted: no cookies, no login, no
workarounds. A site that requires a session simply fails with
URL_FETCH_FAILED, which is the honest outcome.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from urllib.parse import urlparse

import config

logger = logging.getLogger(__name__)


class UrlError(Exception):
    """Carries one of the URL_* contract codes."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class _Deadline:
    """Aborts a download once the wall clock runs out.

    yt-dlp has no overall timeout -- socket_timeout only bounds individual
    reads, so a slow-but-alive server can stream for minutes. Raising from a
    progress hook is the supported way to stop it.
    """

    def __init__(self, seconds: float):
        self.expires_at = time.monotonic() + seconds
        self.seconds = seconds

    def __call__(self, status: dict) -> None:
        if time.monotonic() > self.expires_at:
            raise UrlError(
                "URL_FETCH_FAILED",
                f"Download exceeded the {self.seconds:.0f}s time limit.",
            )


def _looks_like_url(candidate: str) -> bool:
    try:
        parsed = urlparse(candidate.strip())
    except ValueError:
        return False
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def _platform_of(info: dict, url: str) -> str:
    extractor = (info.get("extractor_key") or info.get("extractor") or "").lower()
    if extractor and extractor != "generic":
        return extractor
    return (urlparse(url).netloc or "unknown").removeprefix("www.")


def fetch(url: str, into: Path) -> tuple[Path, dict]:
    """Download one media file into `into`. Returns (path, source_info).

    Raises UrlError with a contract code on every failure path.
    """
    try:
        import yt_dlp
    except ImportError as exc:
        raise UrlError(
            "URL_FETCH_FAILED",
            "URL ingest is unavailable: yt-dlp is not installed.",
        ) from exc

    if not _looks_like_url(url):
        raise UrlError("URL_INVALID", "That is not a valid http(s) URL.")

    probe_options = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "socket_timeout": config.URL_SOCKET_TIMEOUT_S,
        # Never attempt authentication. A login wall is a failure, not a
        # problem to work around.
        "cookiefile": None,
        "cookiesfrombrowser": None,
    }

    # --- 1. probe: duration is checked before a single media byte is fetched
    try:
        with yt_dlp.YoutubeDL(probe_options) as probe:
            info = probe.extract_info(url, download=False)
    except Exception as exc:
        raise UrlError("URL_FETCH_FAILED", _explain(exc)) from exc

    if info is None:
        raise UrlError("URL_FETCH_FAILED", "No media could be found at that URL.")
    if info.get("_type") == "playlist":
        entries = [e for e in (info.get("entries") or []) if e]
        if not entries:
            raise UrlError("URL_FETCH_FAILED", "That playlist contains no media.")
        info = entries[0]

    duration = info.get("duration")
    if duration is not None and duration > config.URL_MAX_DURATION_S:
        raise UrlError(
            "URL_TOO_LONG",
            f"That clip is {int(duration)}s. The limit is "
            f"{config.URL_MAX_DURATION_S}s -- try a shorter one.",
        )

    source = {
        "platform": _platform_of(info, url),
        "title": (info.get("title") or "Untitled")[:200],
        "duration_s": int(duration) if duration is not None else None,
    }

    # --- 2. download, capped on resolution, size and wall clock
    deadline = _Deadline(config.URL_FETCH_TIMEOUT_S)
    download_options = {
        **probe_options,
        # One file, never a merge. `bv*` accepts video-only streams, which is
        # what matters here: YouTube serves no progressive format at or below
        # 720p any more -- every one of them has acodec "none" -- so a selector
        # demanding audio+video finds nothing and fails outright. The pipeline
        # samples frames and ignores audio, so video-only is the correct ask
        # and it avoids needing ffmpeg to mux anything.
        "format": (
            f"bv*[height<={config.URL_MAX_HEIGHT}][ext=mp4]/"
            f"bv*[height<={config.URL_MAX_HEIGHT}]/"
            f"b[height<={config.URL_MAX_HEIGHT}]/"
            f"bv*/b"
        ),
        "max_filesize": config.MAX_FILE_BYTES,
        "outtmpl": str(into / "%(id)s.%(ext)s"),
        "progress_hooks": [deadline],
        "retries": 1,
        "fragment_retries": 1,
        "noprogress": True,
    }

    try:
        with yt_dlp.YoutubeDL(download_options) as downloader:
            downloader.extract_info(url, download=True)
    except UrlError:
        raise
    except Exception as exc:
        raise UrlError("URL_FETCH_FAILED", _explain(exc)) from exc

    files = [p for p in into.iterdir() if p.is_file()]
    if not files:
        # max_filesize aborts by simply not producing a file.
        raise UrlError(
            "URL_TOO_LARGE",
            f"That media exceeds the "
            f"{config.MAX_FILE_BYTES // (1024 * 1024)} MB limit.",
        )

    media = max(files, key=lambda p: p.stat().st_size)
    size = media.stat().st_size
    if size > config.MAX_FILE_BYTES:
        raise UrlError(
            "URL_TOO_LARGE",
            f"That media is {size / 1e6:.0f} MB, over the "
            f"{config.MAX_FILE_BYTES // (1024 * 1024)} MB limit.",
        )

    logger.info("Fetched %s (%s) -> %s, %.1f MB",
                source["platform"], source["title"][:60], media.name, size / 1e6)
    return media, source


def _explain(exc: Exception) -> str:
    """Turn a yt-dlp failure into something a viewer can act on.

    Never leaks a stack trace or a filesystem path.
    """
    text = str(exc).lower()
    # Instagram's wording is "accessible ... without being logged-in" and it
    # suggests passing cookies. We never do that, so surface it as the login
    # wall it is rather than as a generic failure.
    if any(marker in text for marker in (
        "private", "sign in", "log in", "login", "logged-in",
        "cookies", "authentication", "empty media response",
    )):
        return ("That media is private or behind a login wall. Public links "
                "only -- this tool never uses credentials or cookies.")
    if "unavailable" in text or "removed" in text or "deleted" in text:
        return "That media is unavailable, removed, or region-blocked."
    if "unsupported url" in text or "no suitable" in text:
        return "That host is not supported."
    if "timed out" in text or "timeout" in text:
        return "The site did not respond in time."
    if "rate" in text and "limit" in text:
        return "The platform is rate-limiting requests. Try again later."
    return "The media could not be fetched from that URL."
