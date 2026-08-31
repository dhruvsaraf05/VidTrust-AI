"""Provenance signal: EXIF / XMP / C2PA inspection.

Looks for two opposite kinds of evidence:

  * a generator fingerprint (Midjourney, Stable Diffusion, DALL-E, Sora,
    Firefly, or a C2PA / Content Credentials manifest)  -> score 1.0
  * genuine camera EXIF (Make + Model, no generator string)             -> 0.0

If a file carries neither, the signal reports available: False. That case is
common and must not be scored as 0.0, because "no metadata" is not evidence of
authenticity -- every social platform strips EXIF on upload.

This signal is high-precision and low-recall by design: when it fires it is
usually right, and when it stays silent it says so.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable

from PIL import Image, ExifTags

import config

logger = logging.getLogger(__name__)

_TAG_NAMES = {v: k for k, v in ExifTags.TAGS.items()}


def _match_signature(haystack: str, signatures=None) -> str | None:
    """Return the first signature present in the text, if any.

    `signatures` defaults to the full list, which is only safe for decoded
    EXIF/XMP text. Raw binary must pass config.RAW_SCAN_SIGNATURES instead --
    see the note in config.py about short strings matching byte noise.
    """
    lowered = haystack.lower()
    for signature in signatures or config.GENERATOR_SIGNATURES:
        if signature in lowered:
            return signature
    return None


def _describe(signature: str, where: str) -> str:
    """Human-readable detail string, saying where the evidence came from."""
    if signature.startswith("c2pa") or signature in ("jumbf", "content credentials"):
        return f"C2PA / Content Credentials manifest found ({where})"
    return f"Generator signature found: {signature} ({where})"


def _decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def _scan_raw_bytes(path: str) -> str:
    """Read a bounded window from each end of the file as latin-1 text.

    C2PA manifests and XMP packets live near the head of a JPEG/PNG and in the
    moov/meta atoms of an MP4, which sit at one end or the other. Scanning the
    whole file would mean string-searching up to 50 MB per request.
    """
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as handle:
            head = handle.read(config.RAW_SCAN_BYTES)
            if size > config.RAW_SCAN_BYTES * 2:
                handle.seek(-config.RAW_SCAN_BYTES, os.SEEK_END)
                tail = handle.read(config.RAW_SCAN_BYTES)
            else:
                tail = handle.read()
        return (head + tail).decode("latin-1", errors="ignore")
    except Exception as exc:
        logger.warning("Raw metadata scan failed for %s: %s", path, exc)
        return ""


def _collect_image_text(path: str) -> tuple[list[str], bool]:
    """Gather candidate metadata strings from an image.

    Returns (texts, has_camera_exif).
    """
    texts: list[str] = []
    has_camera_exif = False

    try:
        with Image.open(path) as img:
            # PNG tEXt/iTXt chunks -- this is where Automatic1111 / ComfyUI
            # write their full generation parameters.
            for key, value in (img.info or {}).items():
                if isinstance(value, (str, bytes)):
                    texts.append(f"{key}: {_decode(value)}")

            # XMP packet (Adobe Content Credentials, Firefly, many others).
            xmp = (img.info or {}).get("XML:com.adobe.xmp")
            if xmp:
                texts.append(_decode(xmp))

            exif = None
            try:
                exif = img.getexif()
            except Exception:
                exif = None

            if exif:
                make = exif.get(_TAG_NAMES.get("Make", -1))
                model = exif.get(_TAG_NAMES.get("Model", -1))
                if make and model:
                    has_camera_exif = True
                    texts.append(f"Make: {_decode(make)} Model: {_decode(model)}")

                for tag_name in config.EXIF_TEXT_TAGS:
                    tag_id = _TAG_NAMES.get(tag_name)
                    if tag_id is None:
                        continue
                    value = exif.get(tag_id)
                    if value:
                        texts.append(f"{tag_name}: {_decode(value)}")
    except Exception as exc:
        logger.warning("Could not read image metadata from %s: %s", path, exc)

    return texts, has_camera_exif


def analyze(path: str, media_type: str) -> dict:
    """Inspect a file on disk. Returns the signal dict used by the aggregator.

    TODO(roadmap): parse real C2PA manifests with the `c2pa` package so we can
    report the signing authority and validate the claim chain, rather than only
    detecting that a manifest is present. Skipped tonight -- native build.
    TODO(roadmap): video container metadata is currently only a raw byte scan;
    use a real MP4 atom parser (or ffprobe) to read the moov/meta atoms.
    """
    texts: list[str] = []
    has_camera_exif = False

    if media_type == "image":
        texts, has_camera_exif = _collect_image_text(path)

    # --- structured metadata first, with the full signature list -----------
    # These are decoded EXIF tags and XMP packets: real text, where a short
    # generator name is a word rather than a coincidence of bytes.
    for text in texts:
        signature = _match_signature(text)
        if signature:
            return {
                "score": 1.0,
                "available": True,
                "detail": _describe(signature, "metadata"),
            }

    # --- then the bounded raw scan, with the strict list only --------------
    raw = _scan_raw_bytes(path)
    if raw:
        signature = _match_signature(raw, config.RAW_SCAN_SIGNATURES)
        if signature:
            return {
                "score": 1.0,
                "available": True,
                "detail": _describe(signature, "raw scan"),
            }

    # --- genuine camera EXIF ------------------------------------------------
    if has_camera_exif:
        return {
            "score": 0.0,
            "available": True,
            "detail": "Camera EXIF present (Make/Model), no generator signature",
        }

    # --- nothing usable -----------------------------------------------------
    return {
        "score": 0.0,
        "available": False,
        "detail": "No EXIF/XMP/C2PA metadata found (commonly stripped on upload)",
    }
