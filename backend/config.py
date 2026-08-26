"""Central configuration for the VidTrust AI detector backend.

Every tunable number the pipeline depends on lives here so it can be defended
(or corrected) in one place. Constants marked PROVISIONAL have not been fitted
against a labelled dataset yet -- see the evaluation-script work item.
"""

# --------------------------------------------------------------------------
# Signal fusion
# --------------------------------------------------------------------------
# Weights for the weighted average. They sum to 1.0 for the all-signals-present
# case; when a signal is unavailable the aggregator renormalises the remaining
# weights instead of treating the missing signal as a zero score.
WEIGHTS = {
    "model": 0.60,
    "metadata": 0.25,
    "frequency": 0.15,
}

# Human-readable names surfaced in the API response.
SIGNAL_DISPLAY_NAMES = {
    "model": "Classifier",
    "metadata": "Provenance",
    "frequency": "Frequency analysis",
}

# Verdict thresholds on the fused confidence.
THRESHOLD_AI_GENERATED = 0.65   # >= this  -> AI_GENERATED
THRESHOLD_LIKELY_REAL = 0.35    # <= this  -> LIKELY_REAL
                                # in between -> UNCERTAIN

# --------------------------------------------------------------------------
# Upload limits
# --------------------------------------------------------------------------
MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}
ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

# --------------------------------------------------------------------------
# Model signal
# --------------------------------------------------------------------------
MODEL_PRIMARY = "Organika/sdxl-detector"
MODEL_FALLBACK = "umm-maybe/AI-image-detector"

# Label vocabularies. HuggingFace image-classification heads report string
# labels; we map them onto "probability this is machine-generated".
# NOTE: generic heads that report only LABEL_0 / LABEL_1 are deliberately NOT
# mapped here. We cannot know which index means "AI" without checking the model
# card, and guessing would silently invert the signal. An unrecognised
# vocabulary makes the model signal report available: false instead.
AI_LABELS = {"artificial", "ai", "ai_generated", "ai-generated", "fake",
             "generated", "machine", "sdxl", "synthetic"}
REAL_LABELS = {"human", "real", "natural", "photo", "photograph", "authentic"}

# --------------------------------------------------------------------------
# Video sampling
# --------------------------------------------------------------------------
VIDEO_SAMPLE_FPS = 1.0   # target ~1 sampled frame per second of video
VIDEO_MAX_FRAMES = 60    # hard cap so a long clip cannot stall the demo
FREQUENCY_FRAME_STRIDE = 5  # run the FFT signal on every Nth sampled frame

# --------------------------------------------------------------------------
# Frequency signal -- PROVISIONAL CONSTANTS
# --------------------------------------------------------------------------
# This is a weak heuristic, not a trained detector. It measures how much of the
# image's spectral energy sits outside a central low-frequency disc, on the
# folk observation that some generators leave more (upsampling artefacts) or
# less (over-smoothing) high-frequency energy than a camera sensor does.
#
# It is confounded by JPEG quality, resolution, resizing, denoising and image
# content -- a sharp, detailed photograph and an SDXL render can land in the
# same band. It carries the lowest weight for exactly that reason, and the
# numbers below were chosen by inspection, NOT fitted to labelled data.
FFT_IMAGE_SIZE = 512         # square resize before the transform
FFT_LOW_FREQ_RADIUS = 0.03125  # radius of the "low frequency" disc, as a
                               # fraction of FFT_IMAGE_SIZE (=16 px of 512)

# Linear map from the raw high-frequency energy ratio onto 0..1.
# ratio <= FFT_RATIO_MIN -> 0.0, ratio >= FFT_RATIO_MAX -> 1.0.
#
# UNFITTED -- these bound the range where the ratio actually MOVES; they do
# NOT encode any measured AI-vs-real boundary. Measured on synthetic probes at
# radius 0.03125: smooth gradient 0.037, soft blobs 0.183, mid detail 0.835,
# white noise 0.997. A wider low-freq disc (0.125) squashed everything below
# 0.2 and pinned the score at 0.0 for all realistic input, which is why the
# radius is this small.
#
# RECALIBRATE against the 20 real samples before quoting this signal:
#     .venv/Scripts/python calibrate_frequency.py samples/
# Until then the score says "how high-frequency is this image", not "is this
# image AI". Present it that way.
FFT_RATIO_MIN = 0.05
FFT_RATIO_MAX = 0.45

# --------------------------------------------------------------------------
# Metadata signal
# --------------------------------------------------------------------------
# Substrings (matched case-insensitively) that identify a generator or a
# content-credentials manifest.
GENERATOR_SIGNATURES = [
    "midjourney", "stable diffusion", "stablediffusion", "stable-diffusion",
    "sdxl", "dall-e", "dall_e", "dalle", "openai", "sora", "firefly",
    "adobe firefly", "content credentials", "c2pa", "imagen", "veo",
    "flux", "leonardo.ai", "runway", "novelai", "comfyui", "automatic1111",
]

# EXIF tag names inspected for generator strings.
EXIF_TEXT_TAGS = ["Software", "ImageDescription", "Artist", "UserComment",
                  "Copyright", "HostComputer", "XPComment", "XPKeywords"]

# Bytes scanned from each end of a file when looking for raw C2PA/XMP markers.
# Bounded so a 50 MB video does not turn into a 50 MB string search.
RAW_SCAN_BYTES = 512 * 1024
