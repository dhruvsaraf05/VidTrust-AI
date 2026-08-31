"""Materialise a fixed, balanced evaluation set from Community Forensics.

PRD deliverable D4.

Dataset
-------
Community Forensics (small) -- Park & Owens, CVPR 2025.
  https://huggingface.co/datasets/OwensLab/CommunityForensics-Small
  Paper:   https://arxiv.org/abs/2411.04125
  Licence: CC BY-NC-SA 4.0 -- non-commercial research, which is what this is.

Chosen over the alternatives because it is published and citable, it ships
278K real paired with 278K generated images, and every row records the
`model_name` and `architecture` that produced it -- so the failure analysis can
say *which* generators the detector misses, not merely how often it misses.

(The most-downloaded search hit, Hemg/AI-Generated-vs-Real-Images-Datasets, was
rejected: it is CIFAKE merged into an art set and most of its rows are 32x32,
far below what the classifier or the 512-point FFT can use.)

Why this fetches over the datasets-server rather than streaming parquet
----------------------------------------------------------------------
The repository stores the data in 186 parquet shards of 1.2-4 GB each. Measured
throughput on the LFS path was 0.07 MB/s, which puts a single shard at roughly
five hours. The datasets-server /rows endpoint returns the same rows -- all
thirteen columns, with the image bytes inline as base64 -- at about 6 MB/s in
100-row pages. Same data, same labels, minutes instead of hours.

Sampling
--------
Rows are ordered by generator, so consecutive rows come from the same model.
Fake images are therefore capped per generator (--cap) to force spread across
many generators rather than many samples of a few.

The two classes are NOT interleaved: generated images occupy the front of the
indexed range and the real (FFHQ) images sit in a contiguous block at the end,
from roughly offset 8800. A forward-only scan therefore collects its whole fake
quota and never reaches a single real image -- which is exactly what the first
run of this script did. So the fake pass walks forward from the start and the
real pass walks backward from the end, and both run over pages that are almost
entirely the class they want.

Every page is retried with backoff: the datasets-server rate-limits sustained
fetching and drops connections once it does.

NSFW-flagged rows are skipped -- this set goes into a report other people read.

Known limitation, stated rather than hidden
-------------------------------------------
Every real image in this dataset comes from FFHQ: aligned face photographs.
The generated half is varied art and photography. So a detector could separate
the classes partly on CONTENT rather than on synthesis artefacts, which
flatters any result. The summary below prints the real-source breakdown so this
cannot be forgotten when the numbers are written up.

Output
------
    evaluation_set/images/<label>_<nnn>_<slug>.<ext>   original bytes, no re-encode
    evaluation_set/MANIFEST.csv                        ground truth + provenance

Usage
-----
    .venv/Scripts/python build_evaluation_set.py --real 200 --fake 200
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import re
import time
from collections import Counter
from pathlib import Path

import httpx
from PIL import Image

OUT_DIR = Path("evaluation_set")
IMAGES_DIR = OUT_DIR / "images"
MANIFEST = OUT_DIR / "MANIFEST.csv"

ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
DATASET = "OwensLab/CommunityForensics-Small"
CONFIG = "default"
SPLIT = "train"

PAGE_SIZE = 100          # endpoint maximum, and far more efficient than small pages
MAX_OFFSET = 10_500      # the range the datasets-server has indexed
MEASURED_MB_PER_S = 1.9  # observed SUSTAINED rate; bursts hit 6 MB/s but the
                         # datasets-server throttles under continuous fetching
MB_PER_ROW = 0.44        # observed average


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")[:40] or "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=int, default=200)
    parser.add_argument("--fake", type=int, default=200)
    parser.add_argument("--cap", type=int, default=4,
                        help="max images per generator, to force spread")
    parser.add_argument("--resume", action="store_true",
                        help="keep images already on disk and fetch only the shortfall")
    args = parser.parse_args()

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    per_generator: Counter = Counter()
    real_done = fake_done = 0
    scanned = skipped_nsfw = 0
    downloaded_bytes = 0

    if args.resume and MANIFEST.exists():
        with MANIFEST.open(newline="", encoding="utf-8") as handle:
            for existing in csv.DictReader(handle):
                if not (IMAGES_DIR / existing["filename"]).exists():
                    continue
                # CSV gives every field back as a string; the summary sorts
                # these numerically alongside freshly-collected ints.
                existing["width"] = int(existing["width"] or 0)
                existing["height"] = int(existing["height"] or 0)
                rows.append(existing)
                if existing["expected"] == "AI":
                    fake_done += 1
                    per_generator[existing["model_name"]] += 1
                else:
                    real_done += 1
        print(f"resuming : {real_done} real + {fake_done} generated already on disk")
    elif not args.resume:
        for stale in IMAGES_DIR.glob("*"):
            stale.unlink()

    need_real = max(0, args.real - real_done)
    need_fake = max(0, args.fake - fake_done)

    # The real block is almost pure, so those pages are ~100% useful. The fake
    # pass discards most of each page to honour the per-generator cap.
    real_pages = -(-need_real // PAGE_SIZE)
    fake_pages = -(-need_fake // max(1, int(PAGE_SIZE * 0.10))) if need_fake else 0
    projected_mb = (real_pages + fake_pages) * PAGE_SIZE * MB_PER_ROW

    print(f"dataset  : {DATASET}")
    print(f"target   : {args.real} real + {args.fake} generated, "
          f"max {args.cap} per generator")
    print(f"to fetch : {need_real} real + {need_fake} generated "
          f"(~{real_pages + fake_pages} pages, ~{projected_mb:.0f} MB)")
    print(f"estimate : ~{projected_mb / MEASURED_MB_PER_S / 60:.1f} min at the "
          f"measured {MEASURED_MB_PER_S:.1f} MB/s sustained\n")

    client = httpx.Client(timeout=180.0, follow_redirects=True)
    started = time.perf_counter()

    def fetch_page(offset: int):
        """One page, retried with backoff. Returns None if it never arrives."""
        nonlocal downloaded_bytes
        for attempt in range(4):
            try:
                response = client.get(ROWS_ENDPOINT, params={
                    "dataset": DATASET, "config": CONFIG, "split": SPLIT,
                    "offset": offset, "length": PAGE_SIZE,
                })
                downloaded_bytes += len(response.content)
                return response.json().get("rows", [])
            except Exception as exc:
                delay = 4 * (attempt + 1)
                print(f"  ! offset {offset}: {type(exc).__name__}, "
                      f"retry {attempt + 1}/3 in {delay}s")
                time.sleep(delay)
        return None

    def consume(page: list, want_fake: bool) -> None:
        nonlocal scanned, skipped_nsfw, real_done, fake_done
        for item in page:
            row = item.get("row") or {}
            scanned += 1

            if row.get("nsfw_flag"):
                skipped_nsfw += 1
                continue
            label = row.get("label")
            if label is None:
                continue
            is_fake = int(label) == 1
            if is_fake != want_fake:
                continue

            if is_fake:
                if fake_done >= args.fake:
                    return
                generator = row.get("model_name") or "unknown"
                if per_generator[generator] >= args.cap:
                    continue
            elif real_done >= args.real:
                return

            encoded = row.get("image_data")
            if not isinstance(encoded, str):
                continue
            try:
                data = base64.b64decode(encoded)
                with Image.open(io.BytesIO(data)) as probe:
                    width, height = probe.size
                    fmt = (probe.format or "PNG").lower()
                    has_exif = len(probe.info.get("exif", b"")) > 0
            except Exception:
                continue

            kind = "AI" if is_fake else "REAL"
            index = fake_done if is_fake else real_done
            source = row.get("model_name") if is_fake else (row.get("real_source") or "real")
            extension = {"jpeg": "jpg"}.get(fmt, fmt)
            name = f"{kind.lower()}_{index:03d}_{slugify(source)}.{extension}"

            # Original bytes, never re-encoded: re-compression would alter
            # exactly the high-frequency content the FFT signal measures.
            (IMAGES_DIR / name).write_bytes(data)

            rows.append({
                "filename": name,
                "expected": kind,
                "model_name": row.get("model_name") or "",
                "architecture": row.get("architecture") or "",
                "real_source": row.get("real_source") or "",
                "subset": row.get("subset") or "",
                "width": width,
                "height": height,
                "has_exif": "yes" if has_exif else "no",
            })

            if is_fake:
                per_generator[generator] += 1
                fake_done += 1
            else:
                real_done += 1

    def report(offset: int) -> None:
        print(f"  offset {offset:>5}: {real_done:>3} real / {fake_done:>3} generated "
              f"| {downloaded_bytes / 1e6:>4.0f} MB "
              f"| {time.perf_counter() - started:.0f}s", flush=True)

    # --- generated: forward from the front, where they live ----------------
    offset = 0
    while fake_done < args.fake and offset < MAX_OFFSET:
        page = fetch_page(offset)
        if page:
            consume(page, want_fake=True)
            report(offset)
        offset += PAGE_SIZE
        time.sleep(1.0)

    # --- real: backward from the end, where they live ----------------------
    offset = MAX_OFFSET - PAGE_SIZE
    while real_done < args.real and offset >= 0:
        page = fetch_page(offset)
        if page:
            consume(page, want_fake=False)
            report(offset)
        offset -= PAGE_SIZE
        time.sleep(1.0)

    elapsed = time.perf_counter() - started

    if not rows:
        print("\nCollected nothing. The datasets-server may be unavailable.")
        return 1

    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    widths = sorted(r["width"] for r in rows)
    exif_count = sum(1 for r in rows if r["has_exif"] == "yes")
    # For real rows `real_source` is "N/A" -- that column describes which real
    # dataset a GENERATED image was paired with. The actual origin of a real
    # image is in model_name (e.g. FFHQ).
    real_sources = Counter(
        (r["model_name"] or r["real_source"] or "unknown")
        for r in rows if r["expected"] == "REAL"
    )

    print(f"\n{'=' * 68}")
    print(f"collected     : {real_done} real + {fake_done} generated "
          f"= {len(rows)} images")
    print(f"downloaded    : {downloaded_bytes / 1e6:.0f} MB in {elapsed:.0f}s "
          f"({scanned} rows scanned, {skipped_nsfw} NSFW skipped)")
    print(f"generators    : {len(per_generator)} distinct")
    print(f"architectures : {dict(Counter(r['architecture'] for r in rows))}")
    print(f"resolution    : {widths[0]}-{widths[-1]}px wide "
          f"(median {widths[len(widths) // 2]}px)")
    print(f"real sources  : {dict(real_sources)}")
    print(f"with EXIF     : {exif_count} of {len(rows)}")

    if exif_count == 0:
        print("\nNOTE: no image in this set carries EXIF. Public datasets are")
        print("redistributed re-encoded, so the provenance signal cannot fire")
        print("here. This set measures the classifier and frequency signals;")
        print("provenance has to be evaluated on samples/ instead.")

    # A class-separable nuisance variable is worse than a weak signal: it
    # inflates every metric while looking like success. Check the obvious one.
    real_w = {r["width"] for r in rows if r["expected"] == "REAL"}
    fake_w = {r["width"] for r in rows if r["expected"] == "AI"}
    if real_w and fake_w and not (real_w & fake_w):
        print()
        print("WARNING: resolution alone separates the classes -- real images are")
        print(f"{sorted(real_w)}px wide and generated are {sorted(fake_w)}px, with no")
        print("overlap. A detector could exploit that instead of synthesis")
        print("artefacts, so treat the headline figures as an UPPER bound and")
        print("say so in the report.")

    if len(real_sources) == 1:
        only = next(iter(real_sources))
        print(f"\nNOTE: every real image comes from {only}. The real class is")
        print("therefore narrower in content than the generated class, and part")
        print("of any separation may be content rather than synthesis artefacts.")
        print("State this in the report alongside the numbers.")

    print(f"\nwrote {MANIFEST} and {len(rows)} files to {IMAGES_DIR}/")
    print("Next: .venv/Scripts/python evaluate.py --track public")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
