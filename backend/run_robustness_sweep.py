"""Run every condition of the operating-envelope study. One command, in order.

    .venv/Scripts/python run_robustness_sweep.py            # the full sweep
    .venv/Scripts/python run_robustness_sweep.py --dry-run  # print the plan
    .venv/Scripts/python run_robustness_sweep.py --only jpeg50

Each condition is a separate `evaluate.py` invocation, so each writes its own
`evaluation_report_public_<condition>.csv` with the per-signal dump intact.
`robustness.py` then builds the degradation matrix from those CSVs without
re-running inference, exactly as `ablation.py` does for the ablation.

TWO ARMS, AND ONLY ONE OF THEM MEASURES ANYTHING
------------------------------------------------
MATCHED (`--normalise crop --perturb X`) is the measurement. Both classes are
centre-cropped to 512 native pixels first, so a fixed-pixel perturbation is
genuinely the same operation on both.

NATIVE (`--perturb X` alone) is kept deliberately, on three conditions only, as
a demonstration that skipping the control inverts a conclusion. In the public
set resolution IS the label (real 1024, generated 512), and measured on 40+40
images the frequency AUC under sigma=10 noise goes UP on the native arm
(0.8653 -> 0.9012) and DOWN on the matched arm (0.9219 -> 0.8981). Reporting
the native arm alone would claim the signal is noise-robust when noise is
simply correlated with the label.

The clean baselines are the two runs that already exist -- `evaluation_report_
public.csv` (native) and `_normalised.csv` (matched) -- so this script does not
recompute them, and deliberately never passes `--perturb none`, which would
overwrite them.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PYTHON = Path(".venv/Scripts/python")

# (perturbation, normalise) -- order is the order they run in.
MATCHED = ["jpeg95", "jpeg75", "jpeg50", "jpeg30",
           "downscale75", "downscale50",
           "crop256",
           "noise2", "noise5", "noise10",
           "messaging"]

# The confound demonstration. Three is enough to make the point; noise is the
# one that actually inverts, so it is not optional.
NATIVE = ["jpeg50", "noise10", "messaging"]


def conditions() -> list[tuple[str, str]]:
    return ([(p, "crop") for p in MATCHED] + [(p, "none") for p in NATIVE])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", help="run just this perturbation, both arms")
    parser.add_argument("--limit", type=int, default=0,
                        help="pass through to evaluate.py, for a smoke run")
    args = parser.parse_args()

    plan = [c for c in conditions() if not args.only or c[0] == args.only]
    if not plan:
        print(f"No condition matches --only {args.only}")
        return 1

    print(f"{len(plan)} conditions, ~8-9 min each on 400 images "
          f"-> roughly {len(plan) * 8.5 / 60:.1f} h\n")
    for perturb, normalise in plan:
        arm = "matched" if normalise == "crop" else "NATIVE (confound demo)"
        print(f"  {perturb:<12} {arm}")
    print()

    if args.dry_run:
        return 0

    started = time.perf_counter()
    failed: list[str] = []

    for index, (perturb, normalise) in enumerate(plan, start=1):
        command = [str(PYTHON), "evaluate.py", "--track", "public",
                   "--normalise", normalise, "--perturb", perturb]
        if args.limit:
            command += ["--limit", str(args.limit)]

        label = f"{perturb}/{normalise}"
        print(f"[{index}/{len(plan)}] {label} ...", flush=True)
        step = time.perf_counter()
        result = subprocess.run(command, capture_output=True, text=True)
        elapsed = time.perf_counter() - step

        if result.returncode != 0:
            failed.append(label)
            print(f"    FAILED in {elapsed:.0f}s")
            print("    " + (result.stderr or result.stdout).strip()[-400:])
        else:
            # evaluate.py's own summary line is the thing worth keeping.
            auc = next((ln for ln in result.stdout.splitlines()
                        if ln.startswith("roc-auc")), "")
            print(f"    done in {elapsed / 60:.1f} min   {auc.strip()}")

    total = (time.perf_counter() - started) / 60
    print(f"\nsweep finished in {total:.0f} min")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print("next: .venv/Scripts/python robustness.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
