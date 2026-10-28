"""VidTrust AI -- Semester VII backend.

Detects machine-generated media (fully synthetic images and video) by fusing
three independent signals: a pretrained classifier, provenance metadata, and a
frequency-domain heuristic.

Scope note: this is NOT the Semester VI face-swap detector. Nothing here looks
for faces.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager

import shutil
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import config
import reliability
from detectors import frequency_detector, metadata_detector, model_detector, saliency
from pipeline import aggregator
from pipeline.url_fetcher import UrlError, fetch as fetch_url
from pipeline.video_sampler import VideoReadError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("vidtrust")

CHUNK_BYTES = 1024 * 1024


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
class ApiError(Exception):
    """An error that maps onto one of the five contract error codes."""

    def __init__(self, code: str, message: str, status_code: int):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


# Contract error codes -> HTTP status.
#
# NOTE: NO_FACES_DETECTED is carried over from the Semester VI face-swap
# scope. The Semester VII pipeline never inspects faces, so nothing raises it.
# It stays defined so the frozen contract is honoured -- see the handover note.
#
# NOTE: MODEL_UNAVAILABLE is deliberately NOT raised on the analyse path. A
# dead classifier degrades to available: false and the other two signals still
# produce a verdict, which is the whole point of the design. It is reported by
# GET /api/health instead.
ERROR_STATUS = {
    "UNSUPPORTED_FORMAT": 415,
    "FILE_TOO_LARGE": 413,
    "NO_FACES_DETECTED": 422,
    "PROCESSING_FAILED": 500,
    "MODEL_UNAVAILABLE": 503,
    # Additive, for POST /api/analyze-url only. The /api/analyze contract is
    # unchanged and none of these can be returned by it.
    "URL_INVALID": 400,
    "URL_FETCH_FAILED": 502,
    "URL_TOO_LONG": 413,
    "URL_TOO_LARGE": 413,
}


def api_error(code: str, message: str) -> ApiError:
    return ApiError(code, message, ERROR_STATUS[code])


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the classifier once, at startup -- never on the request path."""
    logger.info("Startup: loading classifier (first run downloads weights)...")
    started = time.perf_counter()
    ok = model_detector.load_model()
    elapsed = time.perf_counter() - started
    if ok:
        logger.info("Classifier ready: %s (%.1fs)", model_detector.model_name(), elapsed)
    else:
        logger.error(
            "Classifier NOT available (%.1fs): %s -- the API will still serve "
            "requests using the metadata and frequency signals.",
            elapsed, model_detector.load_error(),
        )
    yield
    logger.info("Shutdown.")


app = FastAPI(
    title="VidTrust AI",
    description="AI-generated media detector (images and video).",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ApiError)
async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.code, "message": exc.message},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Keep FastAPI's own 422 inside the contract shape.

    Without this, a malformed multipart body (most often: the `file` field
    missing or misnamed) returns FastAPI's default {"detail": [...]} envelope,
    which the frontend has no branch for.
    """
    detail = exc.errors()[0].get("msg", "invalid") if exc.errors() else "invalid"

    # The two endpoints take different bodies, so a validation failure has to
    # say which one it is talking about.
    if request.url.path.endswith("/analyze-url"):
        code, message = "URL_INVALID", (
            f"Request body invalid -- send JSON with a single \"url\" field. ({detail})"
        )
    else:
        code, message = "UNSUPPORTED_FORMAT", (
            "Request body invalid -- send multipart/form-data with a "
            f"single field named 'file'. ({detail})"
        )

    return JSONResponse(
        status_code=ERROR_STATUS[code],
        content={"error": code, "message": message},
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Anything we did not anticipate still leaves via the contract shape.

    The client gets the exception TYPE only. Its text used to be included, and
    an OSError's text is routinely a server path (invariant 5); the full
    exception is in the server log.
    """
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "PROCESSING_FAILED",
            "message": f"Unexpected server error ({type(exc).__name__}).",
        },
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health() -> dict:
    """Additive status endpoint. Not part of the frozen /api/analyze contract.

    This is the ONLY place `MODEL_UNAVAILABLE` is reported. When the classifier
    is dead the API is still usable -- the metadata and frequency signals keep
    answering (invariant 6) -- so this deliberately stays HTTP 200 and reports
    the degradation in the body. A 503 would say "cannot serve requests", which
    is not true. The frontend reads `status` for its classifier-down indicator.
    """
    loaded = model_detector.is_available()
    return {
        "status": "ok" if loaded else "degraded",
        "error": None if loaded else "MODEL_UNAVAILABLE",
        "model_loaded": loaded,
        "model_name": model_detector.model_name(),
        "model_error": model_detector.load_error(),
        "weights": config.WEIGHTS,
        "thresholds": {
            "ai_generated": config.THRESHOLD_AI_GENERATED,
            "likely_real": config.THRESHOLD_LIKELY_REAL,
        },
        "max_file_bytes": config.MAX_FILE_BYTES,
        "accepted_extensions": sorted(config.ALLOWED_EXTENSIONS),
        # The link path's real limits, so the interface states them instead of
        # hardcoding a copy that drifts (it said "max 60 s, no cookies" long
        # after both stopped being true).
        "url": {
            "max_duration_s": config.URL_MAX_DURATION_S,
            "max_height": config.URL_MAX_HEIGHT,
            "signed_in_hosts": (sorted(config.URL_COOKIE_HOSTS)
                                if config.URL_COOKIE_FILE or config.URL_BROWSER_COOKIES
                                else []),
        },
    }


async def _spool_upload(upload: UploadFile, suffix: str) -> str:
    """Stream the upload to a temp file, enforcing the size cap as we go.

    Streaming matters: we must not buffer an oversized upload in memory just to
    discover it was oversized.
    """
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    written = 0
    try:
        while True:
            chunk = await upload.read(CHUNK_BYTES)
            if not chunk:
                break
            written += len(chunk)
            if written > config.MAX_FILE_BYTES:
                raise api_error(
                    "FILE_TOO_LARGE",
                    f"File exceeds the {config.MAX_FILE_BYTES // (1024 * 1024)} MB limit.",
                )
            handle.write(chunk)
    except Exception:
        handle.close()
        os.unlink(handle.name)
        raise
    handle.close()

    if written == 0:
        os.unlink(handle.name)
        raise api_error("UNSUPPORTED_FORMAT", "Uploaded file is empty.")

    return handle.name


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)) -> JSONResponse:
    started = time.perf_counter()

    filename = file.filename or "upload"
    extension = os.path.splitext(filename)[1].lower()

    if extension not in config.ALLOWED_EXTENSIONS:
        raise api_error(
            "UNSUPPORTED_FORMAT",
            f"'{extension or filename}' is not supported. Accepted: "
            + ", ".join(sorted(config.ALLOWED_EXTENSIONS)),
        )

    temp_path = await _spool_upload(file, extension)
    is_video = extension in config.VIDEO_EXTENSIONS

    try:
        if is_video:
            result = await run_in_threadpool(aggregator.analyze_video, temp_path)
        else:
            result = await run_in_threadpool(aggregator.analyze_image, temp_path)
    except VideoReadError as exc:
        raise api_error(
            "PROCESSING_FAILED",
            f"Could not read the video: {exc}",
        ) from exc
    except ApiError:
        raise
    except Exception as exc:
        # Log the full exception (with the temp path) server-side, but never
        # leak a server filesystem path back to the client.
        logger.exception("Analysis failed for %s", filename)
        raise api_error(
            "PROCESSING_FAILED",
            f"Could not analyse '{filename}': {type(exc).__name__}. "
            "The file may be corrupt or truncated.",
        ) from exc
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass

    elapsed_ms = int((time.perf_counter() - started) * 1000)

    payload = {
        "id": uuid.uuid4().hex[:6],
        "filename": filename,
        "media_type": result["media_type"],
        "verdict": result["verdict"],
        "confidence": result["confidence"],
        "processing_time_ms": elapsed_ms,
        "signals": result["signals"],
        "frames": result["frames"],
    }
    logger.info(
        "%s -> %s (%.2f) in %dms",
        filename, payload["verdict"], payload["confidence"], elapsed_ms,
    )
    return JSONResponse(content=payload)


class UrlRequest(BaseModel):
    url: str


@app.post("/api/analyze-url")
async def analyze_url(body: UrlRequest) -> JSONResponse:
    """Fetch media from a URL, then run the SAME pipeline an upload runs.

    Additive: /api/analyze is untouched and its contract is unchanged. The
    response here is that contract's shape plus a `source` object.

    There is no analysis code in this path. Everything after the download is
    the existing video pipeline, so a URL and an upload of the same clip give
    the same verdict.
    """
    started = time.perf_counter()
    workspace = Path(tempfile.mkdtemp(prefix="vidtrust_url_"))

    try:
        media_path, source = await run_in_threadpool(fetch_url, body.url, workspace)

        try:
            result = await run_in_threadpool(aggregator.analyze_video, str(media_path))
        except VideoReadError as exc:
            raise api_error(
                "PROCESSING_FAILED",
                f"The media downloaded but could not be read: {exc}",
            ) from exc
        except Exception as exc:
            logger.exception("URL analysis failed for %s", source.get("platform"))
            raise api_error(
                "PROCESSING_FAILED",
                f"Could not analyse the downloaded media: {type(exc).__name__}.",
            ) from exc

        # Provenance is always absent on this path, and the generic wording
        # would understate why: the platform stripped it on upload. Absence of
        # evidence here is the platform's doing, not a property of the media.
        # This rewrites the detail string only -- the score and the
        # availability flag are untouched, so the fusion is unaffected.
        metadata_signal = result["signals"]["metadata"]
        if not metadata_signal["available"]:
            metadata_signal["detail"] = config.URL_METADATA_DETAIL.format(
                platform=source.get("platform", "the platform")
            )

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        payload = {
            "id": uuid.uuid4().hex[:6],
            "filename": source.get("title") or media_path.name,
            "media_type": result["media_type"],
            "verdict": result["verdict"],
            "confidence": result["confidence"],
            "processing_time_ms": elapsed_ms,
            "signals": result["signals"],
            "frames": result["frames"],
            "source": source,
            # Additive, like `source`: the measured reliability for the
            # condition this media arrived in. Computed here because the
            # downloaded file is deleted when the request ends.
            "reliability": reliability.card(str(media_path), "video", source),
        }
        logger.info(
            "url %s (%s) -> %s (%.2f) in %dms",
            source.get("platform"), source.get("title", "")[:50],
            payload["verdict"], payload["confidence"], elapsed_ms,
        )
        return JSONResponse(content=payload)

    except UrlError as exc:
        raise api_error(exc.code, exc.message) from exc
    finally:
        # Same discipline as the upload path: the temp media never outlives
        # the request, whatever happened.
        shutil.rmtree(workspace, ignore_errors=True)


@app.post("/api/explain")
async def explain(file: UploadFile = File(...)) -> JSONResponse:
    """Additive: the evidence behind each signal's reading, for one image.

    `saliency` is the classifier heatmap, `spectrum` the FFT picture the
    frequency score is computed from, `provenance` every metadata field the
    scan read and which one matched.

    Not part of the frozen /api/analyze contract, which is untouched. Images
    only: a per-frame heatmap for video would multiply an already slow request.

    `saliency.available` is false whenever the map is not worth drawing. On
    this checkpoint that is always: the heatmap was measured against occlusion
    and agrees no better than chance (median rho -0.02 over 20 images), so the
    faithfulness gate in detectors/saliency.py declines and says why.
    `saliency.faithfulness` carries the measurement either way.
    """
    filename = file.filename or "upload"
    extension = os.path.splitext(filename)[1].lower()

    if extension not in config.IMAGE_EXTENSIONS:
        raise api_error(
            "UNSUPPORTED_FORMAT",
            f"'{extension or filename}' cannot be explained. Images only: "
            + ", ".join(sorted(config.IMAGE_EXTENSIONS)),
        )

    temp_path = await _spool_upload(file, extension)
    try:
        def run() -> dict:
            from PIL import Image
            with Image.open(temp_path) as image:
                rgb = image.convert("RGB")
            return {
                "saliency": {**saliency.explain(rgb),
                             "faithfulness": saliency.faithfulness()},
                "spectrum": frequency_detector.spectrum(rgb),
                "provenance": metadata_detector.evidence(temp_path, "image"),
            }

        result = await run_in_threadpool(run)
    except Exception as exc:
        logger.exception("Explain failed for %s", filename)
        raise api_error(
            "PROCESSING_FAILED",
            f"Could not explain '{filename}': {type(exc).__name__}. "
            "The file may be corrupt or truncated.",
        ) from exc
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass

    return JSONResponse(content={
        "id": uuid.uuid4().hex[:6],
        "filename": filename,
        **result,
    })


@app.post("/api/reliability")
async def reliability_card(file: UploadFile = File(...)) -> JSONResponse:
    """Additive: how far the measured study says to trust a verdict on this file.

    No model runs. Reads the file's condition (format, size, estimated JPEG
    quality, surviving EXIF), maps it to the nearest condition in
    robustness.json, and quotes what the system achieved there. The frontend
    calls this beside /api/analyze, whose frozen body is untouched.
    """
    filename = file.filename or "upload"
    extension = os.path.splitext(filename)[1].lower()
    if extension not in config.ALLOWED_EXTENSIONS:
        raise api_error(
            "UNSUPPORTED_FORMAT",
            f"'{extension or filename}' is not supported. Accepted: "
            + ", ".join(sorted(config.ALLOWED_EXTENSIONS)),
        )

    temp_path = await _spool_upload(file, extension)
    try:
        media_type = "video" if extension in config.VIDEO_EXTENSIONS else "image"
        result = await run_in_threadpool(reliability.card, temp_path, media_type)
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass

    return JSONResponse(content={"filename": filename, "reliability": result})
