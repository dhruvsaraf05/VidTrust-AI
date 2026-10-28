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

Authentication is scoped, not general. A browser session (or cookie file) is
used only for the hosts in config.URL_COOKIE_HOSTS, which serve nothing to an
anonymous client; every other URL is fetched anonymously. With both
URL_BROWSER_COOKIES and URL_COOKIE_FILE set to None, no credentials are used at
all and a login-walled site fails with URL_FETCH_FAILED. See config.py for why
this was enabled and what it costs.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
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


def _resolves_to_public_host(candidate: str) -> tuple[bool, str]:
    """Reject URLs that point back inside the network. Returns (ok, reason).

    Why this exists
    ---------------
    /api/analyze-url is unauthenticated and hands whatever it is given to
    yt-dlp, which will fetch it. Without this check, a scheme test alone lets
    through http://127.0.0.1:8000/, http://10.x.x.x/ and
    http://169.254.169.254/latest/meta-data/ -- the cloud metadata endpoint.
    The response leaks the hostname back as `source.platform`, and the distinct
    error strings from _explain() distinguish "reachable" from "timed out" from
    "unsupported", which together make a usable blind port scanner. That is a
    server-side request forgery, and it was in this file until it was measured
    and fixed.

    What it cannot do
    -----------------
    This resolves the hostname once and checks every address it gets. It does
    not follow what yt-dlp does afterwards: a permitted host that redirects, or
    serves an HLS manifest pointing somewhere private, is not re-checked, and a
    name that resolves differently on the second lookup (DNS rebinding) would
    slip through. Closing those needs a custom resolver or an egress proxy,
    which is out of scope here. Stated rather than implied.
    """
    host = urlparse(candidate.strip()).hostname
    if not host:
        return False, "no host"

    try:
        resolved = socket.getaddrinfo(host, None)
    except socket.gaierror:
        # A name that does not resolve is not a security decision; let yt-dlp
        # fail on it so the user gets the normal "host not supported" wording.
        return True, "unresolvable, deferred to the fetcher"

    for family, _, _, _, address in resolved:
        if family not in (socket.AF_INET, socket.AF_INET6):
            continue
        try:
            ip = ipaddress.ip_address(address[0])
        except ValueError:
            return False, f"uninterpretable address {address[0]!r}"
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            return False, f"{host} resolves to the non-public address {ip}"

    return True, "public"


def _needs_login(url: str) -> bool:
    """Is this host one that serves nothing without a session?"""
    host = (urlparse(url.strip()).hostname or "").lower().removeprefix("www.")
    return any(host == known or host.endswith("." + known)
               for known in config.URL_COOKIE_HOSTS)


def _cookie_options(url: str) -> dict:
    """Browser cookies, but only for hosts that genuinely require them.

    Scoped deliberately. yt-dlp would happily carry the whole cookie jar to
    whatever host it is pointed at, and this endpoint takes an arbitrary URL
    from an arbitrary user -- so without this gate, pasting a link to a server
    you control would hand it the operator's session cookies. Cookies go to
    config.URL_COOKIE_HOSTS and nowhere else; every other URL is fetched
    anonymously exactly as before.
    """
    if not _needs_login(url):
        return {"cookiefile": None, "cookiesfrombrowser": None}

    # A cookie file wins: it cannot be defeated by a browser lock or by
    # Chromium's app-bound encryption, both of which were measured failing on
    # this machine. See the note in config.py.
    if config.URL_COOKIE_FILE:
        return {"cookiefile": config.URL_COOKIE_FILE, "cookiesfrombrowser": None}

    if config.URL_BROWSER_COOKIES:
        return {"cookiefile": None,
                "cookiesfrombrowser": (config.URL_BROWSER_COOKIES, None, None, None)}

    return {"cookiefile": None, "cookiesfrombrowser": None}


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

    allowed, reason = _resolves_to_public_host(url)
    if not allowed:
        # Logged with the reason, but the client is told only that the target
        # is not permitted: the whole point of the check is to stop this
        # endpoint reporting what is and is not reachable inside the network.
        logger.warning("Refused URL pointing at a non-public host: %s", reason)
        raise UrlError(
            "URL_INVALID",
            "That URL points to a private or local network address, which "
            "this endpoint will not fetch.",
        )

    probe_options = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "socket_timeout": config.URL_SOCKET_TIMEOUT_S,
        # Sets both cookiefile and cookiesfrombrowser, and sets both to None
        # for any host not in config.URL_COOKIE_HOSTS.
        **_cookie_options(url),
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

    # A genuinely private account is a different answer from "we are not signed
    # in", and the user can act on each differently -- so they must not collapse
    # into one message. This branch is only reachable when a session IS being
    # used, because without one Instagram returns the generic empty-response
    # error for everything, public or private alike.
    if "private" in text or "not available to this account" in text:
        return ("That post is from a private account, so it cannot be "
                "fetched. Save the video and upload the file instead.")

    # No session, on a host that will not serve anyone without one.
    if any(marker in text for marker in (
        "sign in", "log in", "login", "logged-in", "cookies",
        "authentication", "empty media response",
    )):
        if config.URL_BROWSER_COOKIES:
            return ("The saved browser session was rejected -- it has most "
                    "likely expired. Sign in to the platform in "
                    f"{config.URL_BROWSER_COOKIES} again, or save the video "
                    "and upload the file instead.")
        return ("This platform serves nothing without a signed-in session, "
                "and no credentials are configured. Save the video and "
                "upload the file instead.")
    if "unavailable" in text or "removed" in text or "deleted" in text:
        return "That media is unavailable, removed, or region-blocked."
    if "unsupported url" in text or "no suitable" in text:
        return "That host is not supported."
    if "timed out" in text or "timeout" in text:
        return "The site did not respond in time."
    if "rate" in text and "limit" in text:
        return "The platform is rate-limiting requests. Try again later."
    return "The media could not be fetched from that URL."
