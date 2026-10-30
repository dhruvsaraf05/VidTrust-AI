# IEEE conference paper

`vidtrust_ieee.tex`: IEEEtran conference format, prepared for ICIP 2027 (IEEE
International Conference on Information Processing, VIT Pune).
`vidtrust_ieee.pdf`: compiled output, **6 pages** (the limit is 4–6).

Builds clean: no errors, no undefined references or citations, no overfull boxes.
Two underfull `\vbox` notices are column-level vertical stretch from balancing
around the full-width figures, and are not visible on the page. All 27 fonts are embedded and none is Type 3, which is what
IEEE PDF eXpress checks.

## Compile it

Locally, with the MiKTeX install used for the committed PDF:

```bash
"C:\Users\HP\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe" -interaction=nonstopmode vidtrust_ieee.tex
```

Run it **twice** so the cross-references resolve. On a first build MiKTeX
fetches IEEEtran and the other packages from CTAN on demand, which takes a few
minutes; after that it is seconds. There is no `.bib`: the bibliography is a
`thebibliography` block inside the `.tex`.

Overleaf works too and needs no local install:

1. New Project → Upload Project → select `vidtrust_ieee.tex`,
   `fig1_architecture.pdf`, `fig4_robustness.pdf` (the `.tex` must be at the
   top level).
2. Set the compiler to **pdfLaTeX** (Menu → Compiler) if it is not already.
3. Recompile. IEEEtran ships with Overleaf, so nothing else is needed.

## Regenerate the figures

```bash
cd paper && python make_figures.py
```

Reads the evaluation outputs in `backend/` and writes four figure PDFs plus PNG
previews. The paper uses `fig1_architecture` and `fig4_robustness`;
`fig2_roc` and `fig3_normalisation` were cut for the page limit and are still
produced for REPORT.md and the slides. Nothing in any figure is typed in by
hand.

## Checks to run before every submission

```bash
cd paper && python audit_numbers.py
cd paper && python order_references.py --check
cd paper && python check_fig1_layout.py
```

- `audit_numbers.py`: 179 assertions that every number in every table and in
  the prose matches the evaluation outputs (`ablation.json`,
  `robustness.json`, `calibration.json`, `saliency_faithfulness.json`, the metrics JSON and per-signal CSVs,
  `failure_analysis.json`, `config.py`). It also asserts the structural facts
  the argument depends on, for example that provenance is still unavailable on
  all 400 Track A files and that two of three uncontrolled conditions still
  change sign. **Run it after any re-evaluation**: the tables are hand-typed
  LaTeX and will otherwise drift from the scripts silently.
- `order_references.py --check`: IEEE numbers references by first citation.
  Without `--check` it reorders the `\bibitem` entries to match, and it refuses
  if any citation lacks an entry or any entry is never cited.
- `check_fig1_layout.py`: measures every label's rendered bounding box in
  Fig. 1 and fails on any overlap or any label spilling out of its box. The
  figure is drawn on an auto-aspect axes, so changing its height crowds the
  text without any error; this is the check that catches it.

## References

17 entries, every one verified against its published record (CVF Open Access,
PMLR, ICLR and NeurIPS proceedings, Springer, IEEE/IRIS repository records).
Venue names use IEEE Reference Guide abbreviations. Two caveats:

- `corvi2023` (ICASSP 2023): no page range could be confirmed anywhere, so the
  entry carries the DOI alone.
- `rossler2019` (FaceForensics++, ICCV 2019): pp. 1–11 comes from a single
  repository record. Worth one check on IEEE Xplore.

## Still to confirm before submitting

- [ ] **Author list and order.** Dhruv Saraf (marked corresponding author), Om
      Mahajan, Arjun Patil, Yash Mahajan, and Dr. Sanjay Vidhani as guide.
      Confirm who is an author, in what order, and that the guide wants to be
      listed. To make the guide the corresponding author, move
      `\IEEEauthorrefmark{1}` to his name and edit the `\thanks` footnote.
- [x] **Venue rules** (VIT's `Author-submission-guidelines.pdf`, read 30 Oct
      2026): IEEE format, 4–6 pages, at most 6 keywords, abstract at most 400
      words, corresponding author clearly marked. The paper has 6 pages, 6
      keywords and a 281-word abstract, and Dhruv is marked. No paper size is
      specified, so it stays US Letter, the IEEE default. Review is not blind,
      since the corresponding author must be named, so the author block stays.
      Submission is through Microsoft CMT at
      https://cmt3.research.microsoft.com/ICIP2027.

## After acceptance (camera-ready)

- Add the IEEE copyright notice line the organisers supply.
- Run the final PDF through IEEE PDF eXpress if the organisers require it.

## Keep labelled as history

Two sets of numbers are quoted as history and are not reproducible from the
current code: the pre-fix provenance false positives (62 of 200) and the
replaced classifier's AUCs (0.7918 / 0.8461). Both are labelled as such in the
text. Keep them labelled.

## What the paper deliberately does not claim

No model was trained or fine-tuned. The fusion weights and verdict thresholds
are unfitted and reported as such, and a fitted fusion that scores better on
held-out data (out-of-fold AUC 0.9674 against 0.9487) is reported and not
adopted. The frequency signal is presented as pointing to content confounding,
not as a synthesis detector. Track A's figures are described as an upper bound,
given the three dataset confounds named in Section VIII.
