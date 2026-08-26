"""Print the raw FFT high-frequency ratio for every image in a folder.

Hour-7 tool. The constants FFT_RATIO_MIN / FFT_RATIO_MAX in config.py were set
to bracket the range where the ratio moves at all, NOT to separate AI from
real -- nothing has been fitted to labelled data.

Use this to set them honestly:

    .venv/Scripts/python calibrate_frequency.py samples/

Then look at where your known-real files cluster versus your known-AI files.
If the two groups overlap heavily, that is a real finding: say so, drop the
frequency weight, and put calibration on the roadmap. Do not tune the numbers
until the demo happens to pass -- that is how you lose a viva.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

import config
from detectors.frequency_detector import _to_score, high_frequency_ratio


def main() -> int:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "samples")
    if not folder.is_dir():
        print(f"Not a directory: {folder}")
        return 1

    files = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in config.IMAGE_EXTENSIONS
    )
    if not files:
        print(f"No images in {folder} (looked for {sorted(config.IMAGE_EXTENSIONS)})")
        return 1

    print(f"{'file':<40} {'raw ratio':>10} {'mapped':>8}")
    print("-" * 60)

    ratios = []
    for path in files:
        try:
            with Image.open(path) as img:
                ratio = high_frequency_ratio(img.convert("RGB"))
        except Exception as exc:
            print(f"{path.name:<40} {'FAILED':>10}  {exc}")
            continue
        ratios.append(ratio)
        print(f"{path.name:<40} {ratio:>10.4f} {_to_score(ratio):>8.3f}")

    if ratios:
        ratios.sort()
        n = len(ratios)
        print("-" * 60)
        print(f"min {ratios[0]:.4f}  median {ratios[n // 2]:.4f}  max {ratios[-1]:.4f}")
        print(f"current scale: MIN={config.FFT_RATIO_MIN} MAX={config.FFT_RATIO_MAX}")
        print(
            "\nIf every mapped score is 0.0 or 1.0, the scale is outside the "
            "data and the signal is contributing nothing but noise."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
