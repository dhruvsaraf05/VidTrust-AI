# PRD — Demo 2 and Evaluation Milestone

**Due:** Thursday 3 September 2026
**Owner:** Dhruv (sole active contributor)
**Status at time of writing:** backend stages 1–5 complete; no frontend,
no labelled samples, no evaluation

---

## 1. Objective

Deliver a working, honestly evaluated AI-generated-media detector, and defend
it in a mentor demo.

The distinguishing claim of this project is not that a detector was built —
many groups will have built one. It is that **the ensemble design was measured
and the contribution of each signal was quantified.** Everything in this
document serves that claim.

### The problem this milestone solves

The system currently works but is unjustified. The fusion weights
(0.60 / 0.25 / 0.15), the verdict thresholds (0.65 / 0.35), and the FFT range
constants (0.05–0.45) were all chosen by inspection. One set of those constants
was already found to be wrong by an order of magnitude — the frequency signal
would have returned 0.0 for essentially every input while still carrying 15% of
the weight.

That is the risk this milestone closes. A viva question of the form *"why 0.6?"*
must have an answer that points at a script.

---

## 2. Success criteria

The milestone succeeds if, on 3 September, all of the following hold.

| # | Criterion | Verifiable by |
|---|---|---|
| S1 | A file can be dragged into a browser and a per-signal verdict returned | live demo |
| S2 | Provenance fires independently on a C2PA-tagged file, visibly | live demo |
| S3 | Video produces a per-frame timeline | live demo |
| S4 | Accuracy, precision, recall, F1 and ROC-AUC exist for a labelled set | `evaluate.py` output |
| S5 | An ablation table shows each signal's contribution | `evaluate.py` output |
| S6 | Thresholds are either derived from the ROC or explicitly justified | report section |
| S7 | A clean clone runs from the README alone | tested on 2 Sept |
| S8 | Every number shown is reproducible by one command | inspection |

S4 and S5 are the ones that matter most. If a day slips, protect them.

---

## 3. Scope

### In scope

- **D1 — Sample set.** ~20 files across the buckets in `samples/README.md`,
  ground truth in `MANIFEST.csv`, swept by `run_samples.py`.
- **D2 — Frequency calibration.** `calibrate_frequency.py` run against real
  media; a decision recorded on whether the signal earns its weight.
- **D3 — Frontend.** `UploadZone`, `VerdictCard`, `SignalPanel`,
  `FrameTimeline`, `ErrorState`. Single page, no routing.
- **D4 — Evaluation set.** 300–500 labelled images, balanced, public,
  disjoint from `samples/`.
- **D5 — `evaluate.py`.** Full metric suite plus per-image per-signal score
  dump.
- **D6 — Ablation study.** Computed from D5's dumped scores.
- **D7 — Threshold selection.** From the ROC curve.
- **D8 — Failure analysis.** The 10 worst misclassifications, examined.
- **D9 — Report section and deck.** Built on D5–D8.

### Explicitly out of scope

Detection history, analytics dashboard, authentication, dark mode, batch
upload, comparison view, model training or fine-tuning, additional detectors,
real C2PA manifest parsing, MP4 atom parsing, video evaluation metrics,
deployment.

**The temptation is to add a fourth signal. Don't.** A fourth unmeasured signal
weakens the project; a measured ablation of three strengthens it.

---

## 4. Deliverable detail

### D1 — Sample set

Follows `samples/README.md`. Non-negotiable properties:

- At least one **AI original with intact provenance metadata**, ideally Adobe
  Firefly (embeds Content Credentials by default). This single file is what
  makes S2 possible. Without it the ensemble thesis is asserted but never
  shown.
- At least 3 real photos with camera EXIF intact, transferred as originals.
  These are the only files exercising the camera-EXIF branch.
- Ground truth may be `UNKNOWN`. A guessed label produces a confidently wrong
  record and is worse than an absent one.

**Acceptance:** `run_samples.py` completes; every file appears in the report
with a verdict or a recorded error.

### D2 — Frequency calibration

Run `calibrate_frequency.py samples/`. Then make an explicit decision and write
it down:

- If real and AI files separate → keep the weight, note the observed margin.
- If they overlap → **set the weight to 0 and report the finding.** This is a
  legitimate result and reads as rigour, not failure.
- Do not retune constants until the demo output looks good. That is fitting to
  the demo, and it is the exact failure this milestone exists to prevent.

**Acceptance:** a decision recorded in the report with the measured spread
behind it.

### D3 — Frontend

Per `FRONTEND_BRIEF_DEMO_1.md`, with the division-of-labour section void.
Additional requirements from `API_NOTES.md`:

- Branch on the `error` field, never `res.status`
- `available: false` renders as "unavailable", never a 0% bar — this
  distinction is the visible form of the design thesis
- Don't build a `NO_FACES_DETECTED` state; it's unreachable
- Progress state must tolerate a 20-second video request
- `signals.*.detail` is free text for display; never parsed

**Acceptance:** S1, S2, S3 demonstrable in a browser.

### D4 — Evaluation set

Constraints, in priority order:

1. **Balanced**, roughly 50/50 real vs AI
2. **Usable resolution** — avoid 32×32 sets; the pipeline resizes to 512 for
   FFT and the classifier needs real detail
3. **Multiple generators** if available; single-generator sets overstate
   performance and that limitation must be stated either way
4. **Public and citable** — it goes in the report bibliography
5. **Capped at 300–500 images** so a full run finishes in minutes on CPU

Any dataset name recalled from memory is unverified until it actually
downloads and loads. Verify before building on it.

**Acceptance:** loads locally, class balance confirmed, source citable.

### D5 — `evaluate.py`

Runs the full pipeline over D4 and writes:

- `evaluation_report.csv` — one row per image: filename, ground truth, fused
  confidence, verdict, and **each signal's raw score and availability**
- `evaluation_metrics.json` — accuracy, precision, recall, F1, ROC-AUC,
  confusion matrix, and counts of `UNCERTAIN` and degraded runs

The per-signal dump is what makes D6 and D7 cheap. Without it, every ablation
variant means re-running inference.

`UNCERTAIN` is recorded as neither correct nor incorrect — it is the system
declining to answer, which is a legitimate third outcome and should be counted
separately rather than forced into the confusion matrix.

**Acceptance:** one command produces both files; rerunning gives identical
numbers.

### D6 — Ablation study

Recomputed from `evaluation_report.csv`, no re-inference.

| Configuration | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Full ensemble | | | | | |
| Model only | | | | | |
| Metadata only | | | | | |
| Frequency only | | | | | |
| Ensemble − metadata | | | | | |
| Ensemble − frequency | | | | | |

Note the confound explicitly: metadata availability correlates with how a file
was obtained, not only with whether it is AI. A metadata-only row will look
strong on originals and collapse on re-saved files. Say so — it's the most
interesting thing in the table.

**Acceptance:** table populated with real numbers, confound noted.

### D7 — Threshold selection

Plot the ROC, choose an operating point, justify it. For a detection system a
false negative typically costs more than a false positive, so deliberately
accepting lower precision is defensible — but say that you chose it, and why.

If the derived thresholds differ from 0.65/0.35, update `config.py` and note
that the API contract's verdict strings are unaffected.

**Acceptance:** thresholds traceable to the curve, or a written reason for
keeping the originals.

### D8 — Failure analysis

The 10 worst misclassifications by margin, looked at individually. Report
patterns: compression level, generator, illustration vs photography,
resolution. This becomes the limitations section and the honest answer when
the mentor probes.

### D9 — Report and deck

Eight slides: reframe · architecture · why an ensemble · live demo ·
evaluation results · **ablation** · limitations · roadmap.

Report section covering methodology, evaluation setup, results, ablation,
failure analysis, limitations.

---

## 5. Schedule

| Date | Deliverables |
|---|---|
| Sat 29 Aug | D1, D2, frontend scaffold |
| Sun 30 Aug | D3 |
| Mon 31 Aug | D4, D5 |
| Tue 1 Sep | D6, D7, D8 |
| Wed 2 Sep | D9, merge to `main`, clean-clone test, rehearsal, backup recording |
| Thu 3 Sep | Demo |

---

## 6. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| No AI file with intact provenance obtainable | S2 lost — the strongest demo moment | Adobe Firefly free tier on 29 Aug; verify EXIF before relying on it |
| Evaluation set won't download or is unusable | S4, S5 lost | Verify 31 Aug morning; fall back to a smaller hand-labelled set of 100 |
| Frontend overruns into Monday | Evaluation compressed | Hard stop Sunday night; ship the UI as-is |
| Frequency signal shows no separation | None — this is a result | Report it, zero the weight, keep the ablation row |
| Classifier performs poorly on the eval set | Awkward but survivable | Report honestly; a measured weak baseline is defensible, an unmeasured strong claim is not |
| Live demo fails on the day | Credibility | 60-second backup recording made 2 Sep |

---

## 7. Demo run order (3 Sept)

1. Real photograph, EXIF intact → low score, provenance reads camera
2. AI image, no metadata → carried by the classifier
3. **AI image with Content Credentials → provenance fires independently**
4. Video → frame timeline
5. A deliberate failure, explained before he finds it
6. Evaluation numbers, then the ablation table

Volunteering the failure in step 5 converts a weakness into evidence of rigour.

---

## 8. Non-goals worth stating aloud

The system does not claim to detect all AI-generated media, does not generalise
to unseen generators without re-evaluation, and does not measure content
quality — the "slop" framing describes the motivation, not the measured
quantity. The measured quantity is provenance: was this machine-generated.

Saying this before the mentor asks is worth more than any accuracy figure.
