"""Hour-7 harness: run every sample through the live API and record what happened.

This is NOT the evaluation script. It computes no accuracy, precision, recall
or F1, and it deliberately prints no aggregate score. Those numbers need a
labelled set and a considered methodology -- that is week 1 work.

What this does: sends each file in samples/ to a running /api/analyze, records
the verdict, confidence, per-signal availability and timing, and flags the ones
that errored or disagreed with the label you wrote in MANIFEST.csv. The point
is to find out which files BREAK the pipeline before the mentor does.

Usage:
    # server must already be running
    .venv/Scripts/python run_samples.py
    .venv/Scripts/python run_samples.py --url http://127.0.0.1:8000 --dir samples
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx

import config

MANIFEST_NAME = "MANIFEST.csv"
VALID_LABELS = {"AI", "REAL", "UNKNOWN"}

# Which verdicts are consistent with which ground-truth label. UNCERTAIN is
# treated as consistent with neither and neither-wrong: it is the system
# declining to answer, which is a legitimate outcome, not a failure.
CONSISTENT = {
    "AI": {"AI_GENERATED"},
    "REAL": {"LIKELY_REAL"},
}


def load_manifest(folder: Path) -> dict[str, dict]:
    """Read MANIFEST.csv if present. Missing manifest is not an error."""
    path = folder / MANIFEST_NAME
    if not path.exists():
        return {}

    entries: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = (row.get("filename") or "").strip()
            if not name or name.startswith("#"):
                continue
            label = (row.get("expected") or "UNKNOWN").strip().upper()
            if label not in VALID_LABELS:
                print(f"  ! {name}: bad expected value {label!r}, treating as UNKNOWN")
                label = "UNKNOWN"
            entries[name] = {
                "expected": label,
                "source": (row.get("source") or "").strip(),
                "notes": (row.get("notes") or "").strip(),
            }
    return entries


def signal_cell(signals: dict, key: str) -> str:
    sig = signals.get(key) or {}
    if not sig.get("available"):
        return "  n/a"
    return f"{sig.get('score', 0.0):5.2f}"


def analyse_one(client: httpx.Client, url: str, path: Path) -> dict:
    """POST one file. Returns a record dict; never raises."""
    try:
        with path.open("rb") as handle:
            response = client.post(
                f"{url}/api/analyze",
                files={"file": (path.name, handle)},
                timeout=180.0,
            )
    except Exception as exc:
        return {"filename": path.name, "ok": False,
                "error": "REQUEST_FAILED", "message": str(exc)}

    try:
        body = response.json()
    except Exception:
        return {"filename": path.name, "ok": False, "error": "BAD_JSON",
                "message": response.text[:200], "status": response.status_code}

    if response.status_code != 200:
        return {"filename": path.name, "ok": False,
                "error": body.get("error", "UNKNOWN"),
                "message": body.get("message", ""),
                "status": response.status_code}

    return {"filename": path.name, "ok": True, "status": 200, **body}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--dir", default="samples")
    parser.add_argument("--out", default="samples_report")
    args = parser.parse_args()

    folder = Path(args.dir)
    if not folder.is_dir():
        print(f"No such directory: {folder}")
        return 1

    files = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in config.ALLOWED_EXTENSIONS
    )
    if not files:
        print(f"No media files in {folder}/. Drop your samples there first.")
        print(f"See {folder}/README.md for what to collect.")
        return 1

    manifest = load_manifest(folder)

    # --- confirm the server is actually up before sending 20 requests -------
    try:
        health = httpx.get(f"{args.url}/api/health", timeout=10.0).json()
    except Exception as exc:
        print(f"Cannot reach {args.url} -- is the server running?\n  {exc}")
        return 1

    print(f"server   : {args.url}")
    print(f"model    : {health.get('model_name')} "
          f"(loaded={health.get('model_loaded')})")
    if not health.get("model_loaded"):
        print(f"  ! classifier NOT loaded: {health.get('model_error')}")
        print("  ! the model signal will report unavailable for every file")
    print(f"files    : {len(files)} in {folder}/")
    print(f"labelled : {sum(1 for f in files if manifest.get(f.name, {}).get('expected') in ('AI', 'REAL'))}"
          f" of {len(files)} have a ground-truth label in {MANIFEST_NAME}")
    print()

    header = (f"{'file':<34}{'exp':>9}{'verdict':>14}{'conf':>7}"
              f"{'model':>7}{'meta':>7}{'freq':>7}{'ms':>7}  flag")
    print(header)
    print("-" * len(header))

    records = []
    errored: list[dict] = []
    inconsistent: list[dict] = []
    degraded: list[dict] = []

    client = httpx.Client()
    for path in files:
        entry = manifest.get(path.name, {})
        expected = entry.get("expected", "UNKNOWN")
        record = analyse_one(client, args.url, path)
        record["expected"] = expected
        record["source"] = entry.get("source", "")
        record["notes"] = entry.get("notes", "")
        records.append(record)

        if not record["ok"]:
            errored.append(record)
            print(f"{path.name[:33]:<34}{expected:>9}{'ERROR':>14}"
                  f"{'-':>7}{'-':>7}{'-':>7}{'-':>7}{'-':>7}  {record['error']}")
            continue

        signals = record.get("signals", {})
        verdict = record.get("verdict", "?")
        confidence = record.get("confidence", 0.0)

        flags = []
        unavailable = [k for k in ("model", "metadata", "frequency")
                       if not (signals.get(k) or {}).get("available")]
        if unavailable:
            flags.append("no:" + ",".join(s[:4] for s in unavailable))
            degraded.append(record)

        if expected in CONSISTENT and verdict not in CONSISTENT[expected]:
            if verdict == "UNCERTAIN":
                flags.append("UNCERTAIN")
            else:
                flags.append("DISAGREES")
                inconsistent.append(record)

        print(f"{path.name[:33]:<34}{expected:>9}{verdict:>14}{confidence:>7.2f}"
              f"{signal_cell(signals, 'model'):>7}"
              f"{signal_cell(signals, 'metadata'):>7}"
              f"{signal_cell(signals, 'frequency'):>7}"
              f"{record.get('processing_time_ms', 0):>7}  {' '.join(flags)}")

    client.close()

    # --- summary: counts of what happened, NOT a score ----------------------
    print()
    print("=" * len(header))
    print(f"ran            : {len(records)} files")
    print(f"errored        : {len(errored)}")
    print(f"ran degraded   : {len(degraded)} (>=1 signal unavailable)")
    print(f"disagreed      : {len(inconsistent)} (labelled file, opposite verdict)")

    if errored:
        print("\nfiles that ERRORED -- fix these first:")
        for record in errored:
            print(f"  {record['filename']:<34} {record['error']}: "
                  f"{record.get('message', '')[:80]}")

    if inconsistent:
        print("\nfiles where the verdict OPPOSED the label:")
        for record in inconsistent:
            print(f"  {record['filename']:<34} expected {record['expected']}, "
                  f"got {record['verdict']} ({record['confidence']:.2f})")

    print("\nNOTE: no accuracy/precision/recall is computed here, by design.")
    print("This harness records behaviour so you can find breakages. Real")
    print("metrics need a properly constructed labelled set -- week 1 work.")

    # --- write the record ---------------------------------------------------
    stamp = datetime.now(timezone.utc).isoformat()
    json_path = Path(f"{args.out}.json")
    json_path.write_text(json.dumps(
        {"run_at": stamp, "url": args.url, "health": health, "results": records},
        indent=2), encoding="utf-8")

    csv_path = Path(f"{args.out}.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "expected", "media_type", "verdict",
                         "confidence", "model", "metadata", "frequency",
                         "ms", "error"])
        for record in records:
            signals = record.get("signals", {})

            def cell(key: str) -> str:
                sig = signals.get(key) or {}
                return "" if not sig.get("available") else f"{sig.get('score', 0.0):.4f}"

            writer.writerow([
                record["filename"], record.get("expected", ""),
                record.get("media_type", ""), record.get("verdict", ""),
                record.get("confidence", ""), cell("model"), cell("metadata"),
                cell("frequency"), record.get("processing_time_ms", ""),
                "" if record["ok"] else record.get("error", ""),
            ])

    print(f"\nwrote {json_path} and {csv_path}")
    return 1 if errored else 0


if __name__ == "__main__":
    raise SystemExit(main())
