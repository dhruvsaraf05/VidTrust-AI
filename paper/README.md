# IEEE paper draft

`vidtrust_ieee.tex` — first full draft, IEEEtran conference format.
`vidtrust_ieee.pdf` — compiled output, 7 pages.

Builds clean: no errors, no undefined references or citations, no overfull
boxes. Four underfull-hbox warnings remain (loose inter-word spacing in
justified two-column text); they are cosmetic and normal for this class.

## Compile it

Locally, with the MiKTeX install used for the committed PDF:

```bash
"C:\Users\HP\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe" -interaction=nonstopmode vidtrust_ieee.tex
```

Run it **twice** so the cross-references resolve. On a first build MiKTeX
fetches IEEEtran and the other packages from CTAN on demand, which takes a few
minutes; after that it is seconds. There is no `.bib` — the bibliography is a
`thebibliography` block inside the `.tex`.

Overleaf works too and needs no local install:

1. New Project → Upload Project → select `vidtrust_ieee.tex`,
   `fig1_architecture.pdf`, `fig2_roc.pdf`, `fig3_normalisation.pdf`
   (drag all four in; the `.tex` must be at the top level).
2. Set the compiler to **pdfLaTeX** (Menu → Compiler) if it is not already.
3. Recompile. IEEEtran ships with Overleaf, so nothing else is needed.

## Regenerate the figures

```bash
cd paper && python make_figures.py
```

Reads `backend/evaluation_report_public*.csv` and `backend/ablation.json` and
writes the three PDFs (plus PNG previews for quick viewing). Nothing in the
figures is typed in by hand.

## Check the numbers still match

```bash
cd paper && python audit_numbers.py
```

91 assertions covering every figure in every table and the derived claims in
the prose, checked against `ablation.json`, `evaluation_metrics_public*.json`,
`failure_analysis.json`, the per-signal CSVs and `config.py`. **Run this after
any re-evaluation** — the tables are hand-typed LaTeX and will otherwise drift
from the scripts silently. It also asserts the structural facts the paper's
argument depends on: that provenance is still unavailable on all 400 Track A
files, that the two identity rows in the ablation are still identities, and
that the centre crop still leaves generated-image scores untouched.

## Before this goes anywhere

- [ ] **Author list.** Taken from the project deck's title slide: Dhruv Saraf,
      Om Mahajan, Arjun Patil, Yash Mahajan, with Dr. Sanjay Vidhani as guide.
      Confirm who is actually an author, in what order, and whether the guide
      is a co-author. Yash Mahajan's roll number on the deck is 10 digits where
      the others are 11 — check it.
- [ ] **References.** All twelve are real papers, but the entries here carry no
      pages, DOIs or publisher detail, and were written from memory rather than
      fetched. Verify each against the published record before submission.
      `park2025` (Community Forensics) is the one most worth double-checking:
      it is the dataset the whole evaluation rests on.
- [ ] **Venue, and then length.** The draft is ~4,100 words of body text plus
      3 figures, 4 tables and 12 references, which lands around 7 pages in
      IEEEtran two-column — over a typical 6-page limit. Section V (Results)
      and Section VII (Limitations) are the natural places to cut. The abstract
      is 238 words; some venues cap it at 150–200.
- [ ] **Anonymisation.** If the venue is double-blind, strip the author block
      and the repository references.
- [ ] **Historic figures.** Two sets of numbers are quoted as history and are
      not reproducible from the current code: the pre-fix provenance false
      positives (62 of 200) and the replaced classifier's AUCs
      (0.7918 / 0.8461). Both are labelled as such in the text. Keep them
      labelled.

## What the draft deliberately does not claim

No model was trained or fine-tuned. The fusion weights and verdict thresholds
are unfitted and are reported as such, including a measured operating point
that improves held-out accuracy and was not adopted. The frequency signal is
presented as unattributed, not as a synthesis detector. Track A's figures are
described as an upper bound, given the three dataset confounds named in
Section VII.
