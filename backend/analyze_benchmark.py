"""Rank the benchmarked classifiers, and test whether combining two helps.

Reads classifier_benchmark.csv. Selection is made on the public set -- the
400-image Community Forensics subset built for exactly this -- and `samples`
is reported alongside as an independent check, never as the thing chosen on
(invariant 10: never tune on the demo files).

    .venv/Scripts/python analyze_benchmark.py
"""

from __future__ import annotations

import csv
import itertools
from collections import defaultdict
from pathlib import Path

from evaluate import roc_auc


def load():
    scores = defaultdict(dict)      # (set, model) -> {filename: score}
    truth = defaultdict(dict)       # set -> {filename: 1/0}
    with Path("classifier_benchmark.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            scores[(row["set"], row["model"])][row["filename"]] = float(row["score"])
            truth[row["set"]][row["filename"]] = 1 if row["expected"] == "AI" else 0
    return scores, truth


def metrics(pairs):
    """pairs: list of (score, label). Returns AUC and accuracy at a 0.5 cut."""
    if not pairs:
        return None, None, None, None
    s = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    auc = roc_auc(s, y)
    acc = sum(1 for sc, yy in pairs if (sc >= 0.5) == bool(yy)) / len(pairs)
    ai = [sc for sc, yy in pairs if yy == 1]
    real = [sc for sc, yy in pairs if yy == 0]
    mean_ai = sum(ai) / len(ai) if ai else None
    mean_real = sum(real) / len(real) if real else None
    return auc, acc, mean_ai, mean_real


def show(title, rows):
    print(f"\n{title}")
    print(f"{'configuration':<50}{'AUC':>8}{'acc@.5':>9}{'meanAI':>9}{'meanREAL':>10}")
    print("-" * 86)
    for name, auc, acc, m_ai, m_real in rows:
        f = lambda v: "  --  " if v is None else f"{v:.4f}"
        print(f"{name[:49]:<50}{f(auc):>8}{f(acc):>9}{f(m_ai):>9}{f(m_real):>10}")


def main() -> int:
    scores, truth = load()
    models = sorted({m for (_, m) in scores})
    print(f"{len(models)} models benchmarked")

    for set_name in ("public", "samples"):
        y = truth.get(set_name, {})
        if not y:
            continue
        rows = []
        for m in models:
            sc = scores.get((set_name, m), {})
            pairs = [(sc[f], y[f]) for f in y if f in sc]
            rows.append((m, *metrics(pairs)))
        rows.sort(key=lambda r: -(r[1] or 0))
        show(f"=== {set_name.upper()}  (n={len(y)}) — single models", rows)

        # Pairwise combinations. mean() is a soft vote; max() fires if EITHER
        # model is confident, which is the behaviour wanted when each model
        # covers a different generator family.
        combos = []
        for a, b in itertools.combinations(models, 2):
            sa, sb = scores.get((set_name, a), {}), scores.get((set_name, b), {})
            common = [f for f in y if f in sa and f in sb]
            if not common:
                continue
            for op, fn in (("mean", lambda x, z: (x + z) / 2),
                           ("max", max)):
                pairs = [(fn(sa[f], sb[f]), y[f]) for f in common]
                combos.append((f"{op}({a.split('/')[-1]}, {b.split('/')[-1]})",
                               *metrics(pairs)))
        combos.sort(key=lambda r: -(r[1] or 0))
        show(f"=== {set_name.upper()} — best pairings", combos[:10])

    print("\nSelection is made on PUBLIC. `samples` is the independent check;")
    print("choosing on it would be fitting to the demo files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
