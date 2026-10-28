"""Image perturbations for the operating-envelope study.

Every function here is a pure PIL -> PIL transform. No file IO, no config
reads, no randomness that is not seeded: a condition must reproduce exactly on
a re-run or the degradation matrix is not a measurement.

WHY THESE CONDITIONS
--------------------
They are the transforms real media actually undergoes between being created and
being checked: it gets re-saved as JPEG, resized by a platform, screenshotted,
or pushed through a messaging app that does several of those at once. A detector
that only works on pristine originals is not a detector of anything useful, and
until this module existed we had no measured answer to "what happens to a
WhatsApp image?" -- the single most likely question about this system.

THE CONFOUND THESE MUST BE RUN AGAINST
--------------------------------------
In the public evaluation set, resolution IS the label: all 200 real images are
1024x1024 and all 200 generated are 512x512, with no overlap. So a perturbation
defined in absolute pixels is partly a class indicator, and the measurement can
inflate or even invert.

Measured, on 40+40 images, frequency-signal AUC under Gaussian noise:

    sigma   native resolution   matched 512 crop
    0       0.8653              0.9219
    2       0.8647              0.9209
    5       0.8697              0.9141
    10      0.9012              0.8981     <- native goes UP, matched goes DOWN

On the native set, adding noise *improves* apparent separation, because
frequency_detector resizes to 512 before measuring: a 1024 image has its noise
averaged 2x2 away, a 512 image does not. Report that and you have published
"the frequency signal is noise-robust" when the truth is "noise correlates with
the label".

So the matched arm (--normalise crop --perturb X) is the measurement, and the
native arm (--perturb X alone) is kept deliberately as a demonstration of the
confound. See REPORT.md section 6 for the same argument about resampling.

ORDER MATTERS
-------------
evaluate.py applies crop THEN perturb. JPEG-of-crop and crop-of-JPEG differ --
for JPEG, by whether the 8x8 block grid aligns to the crop origin. The order is
recorded in the metrics JSON so a run can be reproduced.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

# A fixed seed per image would make noise identical across images; a fixed seed
# per *run* makes the whole condition reproducible while still varying the noise
# field between images. Seeded at module import, used by the noise conditions.
_NOISE_SEED = 20260903


def _jpeg(image: Image.Image, quality: int) -> Image.Image:
    """Round-trip through JPEG at a given quality, in memory."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


def _downscale_restore(image: Image.Image, factor: float) -> Image.Image:
    """Shrink by `factor` then scale back to the original size.

    This is what a platform does when it serves a thumbnail that someone then
    saves: the pixel count comes back but the detail does not. Both legs use
    LANCZOS, so the loss is the resampling itself rather than a crude filter.
    """
    width, height = image.size
    small = image.resize(
        (max(1, round(width * factor)), max(1, round(height * factor))),
        Image.LANCZOS,
    )
    return small.resize((width, height), Image.LANCZOS)


def _centre_crop(image: Image.Image, size: int) -> Image.Image:
    """Take `size` native pixels from the centre. Nothing is resampled.

    Upscales first only if the image is smaller than the target, which does
    resample those files -- but in this study the input is already a 512 crop,
    so for size <= 512 that branch never fires.
    """
    width, height = image.size
    if width < size or height < size:
        scale = size / min(width, height)
        image = image.resize(
            (max(size, round(width * scale)), max(size, round(height * scale))),
            Image.LANCZOS,
        )
        width, height = image.size
    left, top = (width - size) // 2, (height - size) // 2
    return image.crop((left, top, left + size, top + size))


def _gaussian_noise(image: Image.Image, sigma: float) -> Image.Image:
    """Add zero-mean Gaussian noise of standard deviation `sigma` (0-255)."""
    rng = np.random.default_rng(_NOISE_SEED)
    array = np.asarray(image.convert("RGB"), dtype=np.float64)
    noisy = array + rng.normal(0.0, sigma, array.shape)
    return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8), mode="RGB")


def _messaging_app(image: Image.Image) -> Image.Image:
    """Approximate what a messaging platform does on upload.

    Resize the long edge to 1080 IN EITHER DIRECTION, then JPEG ~70.

    The "in either direction" is the whole point and was a bug in the first
    draft of this study. Both classes in the public set are already <= 1080, so
    a downscale-only rule never fires: the condition silently collapses to
    "JPEG 70", and you report a resize result that never happened. Verified --
    output sizes were unchanged at {(512,512), (1024,1024)} and the frequency
    AUC was identical to clean at 4 d.p.

    Note that on the NATIVE set the two classes still get different scale
    factors (512 -> 1080 is 2.11x up, 1024 -> 1080 is 1.05x up), which is
    exactly the confound described in the module docstring. On the matched 512
    crops both get the same 2.11x, which is why the matched arm is the one that
    measures anything.
    """
    width, height = image.size
    scale = 1080 / max(width, height)
    resized = image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.LANCZOS,
    )
    return _jpeg(resized, 70)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
PERTURBATIONS = {
    "none": (lambda image: image, "no perturbation (baseline)"),

    "jpeg95": (lambda image: _jpeg(image, 95), "JPEG re-encode, quality 95"),
    "jpeg75": (lambda image: _jpeg(image, 75), "JPEG re-encode, quality 75"),
    "jpeg50": (lambda image: _jpeg(image, 50), "JPEG re-encode, quality 50"),
    "jpeg30": (lambda image: _jpeg(image, 30), "JPEG re-encode, quality 30"),

    "downscale75": (lambda image: _downscale_restore(image, 0.75),
                    "downscale to 75% and restore (LANCZOS both ways)"),
    "downscale50": (lambda image: _downscale_restore(image, 0.50),
                    "downscale to 50% and restore (LANCZOS both ways)"),

    "crop256": (lambda image: _centre_crop(image, 256),
                "centre crop to 256x256 native pixels"),

    "noise2": (lambda image: _gaussian_noise(image, 2.0),
               "additive Gaussian noise, sigma 2/255"),
    "noise5": (lambda image: _gaussian_noise(image, 5.0),
               "additive Gaussian noise, sigma 5/255"),
    "noise10": (lambda image: _gaussian_noise(image, 10.0),
                "additive Gaussian noise, sigma 10/255"),

    "messaging": (_messaging_app,
                  "long edge to 1080 (either direction) then JPEG 70"),
}

# Every perturbation except "none" re-encodes the pixels and therefore destroys
# EXIF/XMP. evaluate.py refuses the samples track for any of these, for the same
# reason it already refuses --normalise crop there: that track exists to
# exercise provenance, and a condition that strips provenance measures nothing.
LOSSY = tuple(name for name in PERTURBATIONS if name != "none")


def apply(name: str, image: Image.Image) -> Image.Image:
    """Apply the named perturbation. Raises KeyError on an unknown name."""
    return PERTURBATIONS[name][0](image)


def describe(name: str) -> str:
    """Human-readable description, recorded in the metrics JSON."""
    return PERTURBATIONS[name][1]
