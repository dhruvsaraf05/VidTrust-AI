# VidTrust AI — Evaluation Report

Semester VII, B.Tech (Information Technology), Somaiya Vidyavihar University.

Every figure below is produced by a script in this repository. The commands
that generate each one are listed in [README.md](README.md#reproducing-every-number).
No figure from the Semester VI report appears anywhere in this document.

---

## 1. Scope change from Semester VI

Semester VI built a deepfake **face-swap** detector: XceptionNet fine-tuned on
FaceForensics++, with a FastAPI backend and a React frontend. It was submitted
and defended.

Semester VII is a different system with a different threat model. The dominant
form of manipulated media is no longer a face swapped into an existing video;
it is media that is **synthetic in its entirety** — an image or clip that was
never captured by any camera. Detecting a swapped face is irrelevant to an
image that contains no photographed face at all.

So the pipeline was rebuilt. Nothing in it inspects faces: there is no face
detector, no MTCNN, no facial crop, no XceptionNet. The name "VidTrust AI" is
retained for continuity; the scope is not.

One consequence is visible in the API contract. The error code
`NO_FACES_DETECTED` survives from the Semester VI contract and is now
permanently unreachable — no code path can raise it. It is kept so the frozen
contract is honoured, and documented as dead rather than quietly removed.

## 2. Ensemble architecture and the renormalisation rule

Three signals, deliberately independent in the evidence they use:

| Signal | Key | Weight | Evidence |
|---|---|---|---|
| Classifier | `model` | 0.60 | `Organika/sdxl-detector`, inference only, no fine-tuning |
| Provenance | `metadata` | 0.25 | EXIF / XMP / C2PA generator fingerprints |
| Frequency | `frequency` | 0.15 | FFT high-frequency energy ratio |

The classifier is loaded once at application startup and never on the request
path. No model in this project was trained or fine-tuned.

**The renormalisation rule.** A signal that cannot run is excluded from the
weighted average and the remaining weights are rescaled:

```
confidence = Σ(score × weight) / Σ(weight)     over AVAILABLE signals only
```

A missing signal is never scored 0.0. That distinction is the single most
important behaviour in the system, because 0.0 means "this looks real" while
absence means "no evidence either way", and conflating them lets a stripped
file drift toward an unearned verdict of authentic. When metadata is absent,
the classifier and frequency signals are rescaled from 0.60/0.15 to 0.80/0.20
and decide between themselves.

If no signal at all is available the confidence is held at 0.500 — maximum
ignorance, which lands inside the UNCERTAIN band — rather than defaulting to a
reading in either direction.

`UNCERTAIN` is a first-class outcome, not a hedge. It means the system declines
to answer, and throughout this report it is counted separately from correct and
incorrect rather than forced into a confusion matrix.

## 3. Evaluation methodology

### Why there are two tracks

**Track A — public set**, 400 images from Community Forensics (small), Park &
Owens, CVPR 2025, CC BY-NC-SA 4.0. 200 real / 200 generated across 50 distinct
generators.

**Track B — hand-collected set**, 16 files in `samples/`, 13 labelled.

They are never averaged. Public datasets are redistributed re-encoded, so
**Track A carries no EXIF at all** and the provenance signal cannot fire on a
single one of its 400 images. Averaging the tracks would let 400 images with a
structurally dead signal drown 13 that exercise it, and the reported "ensemble"
headline would silently be a two-signal result. Track A measures the classifier
and frequency signals at scale; Track B is the only place provenance does
anything.

Track B is additionally **not a held-out estimate** — those files were used
during development and are the demo material. Its figures are a sanity check.

### The selection / reporting split

Thresholds are chosen on one half of Track A and reported on the other, split
stratified by class and seeded. A threshold picked on the same rows it is then
scored against is fitted to those rows, and the resulting number estimates
nothing.

This was not a formality. An earlier threshold search over all 400 rows at once
suggested a single cut near 0.06 lifting accuracy to roughly 0.73. Under a
proper split on the same (naive) data the selection half chose a completely
different cut, 0.806, which reached only 0.6850 on the held-out half. The
apparent gain was an artefact of choosing and reporting on the same data.

## 4. Track A results

### 4.1 Headline

| | naive | **normalised** |
|---|---|---|
| accuracy | 0.6532 | **0.7302** |
| precision | 0.6705 | **0.8112** |
| recall | 0.6170 | 0.6170 |
| F1 | 0.6427 | **0.7009** |
| ROC-AUC | 0.7918 | **0.8461** |
| coverage | 0.930 | 0.9175 |
| abstained | 28 | 33 |

Confusion matrix, normalised condition, positive class = AI, 33 abstentions
excluded:

| | predicted AI | predicted REAL |
|---|---|---|
| **actual AI** | 116 | 72 |
| **actual REAL** | 27 | 152 |

Signal availability across all 400: model 400, **metadata 0**, frequency 400.
Every run is therefore "degraded" by definition.

### 4.2 Ablation

Recomputed from the per-signal scores dumped by `evaluate.py`; no inference is
re-run. Each configuration re-fuses using the same renormalisation rule the
live pipeline uses, so dropping a signal is exactly equivalent to that signal
being unavailable at runtime.

| configuration | naive acc | naive AUC | norm acc | norm AUC |
|---|---|---|---|---|
| full ensemble | 0.6532 | 0.7918 | 0.7302 | 0.8461 |
| model only | 0.6464 | 0.7064 | 0.7135 | 0.7482 |
| metadata only | — | 0.5000 | — | 0.5000 |
| frequency only | 0.7194 | 0.9012 | 0.7177 | 0.9129 |
| ensemble − metadata | 0.6532 | 0.7918 | 0.7302 | 0.8461 |
| ensemble − frequency | 0.6464 | 0.7064 | 0.7135 | 0.7482 |

**The table has six rows but only three distinct results.** Because metadata is
available on 0 of 400 files, "ensemble − metadata" is arithmetically identical
to the full ensemble, and "ensemble − frequency" is identical to model-only.
These are not independent findings and must not be presented as such — they are
identities forced by the dataset, and they are the clearest argument for why
Track B exists.

The `metadata only` row cannot produce accuracy, precision, recall or F1 at
all: with no signal available every image abstains, so there are zero decided
cases. Its AUC of 0.5000 is the definition of a constant predictor, not a
measurement.

Frequency-only is lopsided in both conditions: **precision 1.0000, recall
0.3237**. It never produces a false positive and catches roughly a third of
generated images.

### 4.3 Threshold selection

| condition | operating point | chosen on selection | **on held-out half** | coverage (held-out) |
|---|---|---|---|---|
| naive | single cut 0.806 | 0.7150 | 0.6850 | 1.00 |
| naive | band 0.06 / 0.70 | 0.7786 | 0.7681 | 0.69 |
| naive | current 0.35 / 0.65 | — | 0.6170 | 0.94 |
| normalised | single cut 0.060 | 0.7800 | 0.7500 | 1.00 |
| normalised | band 0.06 / 0.68 | 0.8741 | **0.8492** | 0.63 |
| normalised | current 0.35 / 0.65 | — | 0.6885 | 0.92 |

**The measured band was not adopted.** `config.py` still holds 0.35 / 0.65.

The result is real and survives the split: 0.8492 against 0.6885 on data never
used to choose it. It was declined for two reasons. First, it costs **29 points
of coverage** — abstaining on 37% of inputs instead of 8% — and a detector that
declines to answer on more than a third of what it is shown is a different
product, not a tuned one. Second, it is fitted to a score distribution shaped by
the dataset confounds in §5, on a set of face photographs versus digital art,
which is not what the demonstration inputs look like.

Adopting it would be trading a measurable gain on a confounded set for unknown
behaviour on real inputs. The finding is recorded; the constants are not
changed.

## 5. Dataset limitations

Three properties of the public set inflate its results. All three are detected
and printed by `build_evaluation_set.py` on every run.

**Zero EXIF.** Not one of the 400 images carries EXIF. Public datasets are
redistributed re-encoded, which strips it. The provenance signal is structurally
dead on Track A — hence the two-track design.

**FFHQ content confound.** Every real image comes from FFHQ: aligned,
centred face photographs. The generated half is varied digital art,
illustration and photography. A detector can separate these partly on *subject
matter* rather than on synthesis artefacts, and nothing in the metrics
distinguishes the two.

**Resolution asymmetry.** Real images are natively 1024×1024; generated images
are natively 512×512, with **no overlap**. Since every detector resizes to 512
internally, real images arrived having been downsampled — which destroys
high-frequency content — while generated images arrived untouched. This is
addressed in §6.

Together these mean **Track A's figures are an upper bound**, not an estimate of
field performance.

## 6. Normalisation: why centre-crop, not resize

Resizing every image to a common target is the obvious control and it does not
work. A 1024×1024 image resized to 512 is still downsampled 2×, while a native
512 image is not. The *operation* is identical but its *effect* depends on input
size — which is precisely the variable that differs by class. The confound
survives the fix.

A **centre crop of 512×512 native pixels** interpolates nothing. No frequency
content is created or destroyed, and every image reaches the detectors as an
unresampled 512×512 block, so the processing history is genuinely identical
across classes. Crops are written as PNG (lossless) so the control itself adds
no compression artefacts.

What it does not fix: a crop shows a sub-region of a large image and the whole
of a small one, so framing still differs. That is a content difference, not a
resampling one, and it is reported as such. Normalisation is refused on Track B,
because re-saving as PNG discards the EXIF that track exists to test.

---

## 7. Named finding (a): the metadata false-positive bug

**This section is the argument for why evaluation is not optional.**

`metadata_detector` scans a bounded window of raw file bytes for generator
fingerprints. The signature list contained **`"veo"` — three characters**. In a
few hundred kilobytes of compressed image data, a three-character
case-insensitive byte sequence occurs by chance roughly a quarter of the time.

The result: the provenance signal reported **"Generator signature found: veo"
at score 1.0 — full confidence — on 62 of 200 real photographs.** Across the
set it fired on 65 real images against 23 generated ones. The signal intended
to prove synthetic origin was firing *more often on real photographs than on
generated images*, and contributing a quarter of the fused weight toward
AI_GENERATED every time it did.

Three things are worth drawing out.

**It was invisible without measurement.** The API returned well-formed
responses. The detail string was plausible. Every unit-level behaviour was
correct: the scan found the string it was asked to find. Nothing short of
running the detector over a labelled set of known-real images and reading the
per-signal dump would have surfaced it. It was found because the evaluation
harness records each signal's score *and* its availability flag per file, not
just the fused verdict.

**The fix is a separation of concerns, not a deletion.** Signature matching now
uses two lists: the full list for decoded EXIF and XMP text, where a short
generator name is a word; and a strict list requiring six or more characters,
plus structural C2PA box markers, for raw binary. Verified afterwards: 0 of 400
false positives, while a genuine EXIF `Software: Midjourney v6` tag is still
detected.

**It has a stated cost.** A video whose only evidence is the bare string
"sora" in its container is no longer detected. That trade is deliberate and
documented in `config.py`: a signal that fires on a third of all real
photographs is worse than one with a known blind spot.

## 8. Named finding (b): the ensemble underperforms its best component

Under normalisation, **frequency-only reaches AUC 0.9129 while the full fused
ensemble reaches 0.8461.** The ensemble is worse than one of its own
constituents, and that constituent carries the smallest weight in the fusion.

Stated plainly: on this dataset, the hand-chosen weights make the system worse
than using its cheapest signal alone.

Two things follow, and they pull in opposite directions.

**The weights are unfitted.** 0.60 / 0.25 / 0.15 were chosen by inspection
before any measurement existed. There is no evidence behind them, they were
never claimed to be optimal, and this result is what an unfitted prior looks
like when it is finally measured.

**But reweighting toward frequency would be fitting to a confound.** The
resolution asymmetry of §5 was the obvious explanation, and it was tested: under
normalisation, which removes the resampling difference entirely, frequency-only
AUC did not collapse — it rose slightly, from 0.9012 to 0.9129. So resampling
was not the cause.

That is a genuine finding, and it is still not evidence that the frequency
signal detects synthesis. A content confound remains, and the crop plausibly
*intensified* it: a 512×512 centre crop of a 1024×1024 FFHQ portrait is mostly
smooth skin, while the generated images are whole detailed scenes. Smooth versus
detailed is exactly what an FFT high-frequency ratio measures. Frequency-only
precision of 1.000 at recall 0.324 is consistent with a signal that fires
confidently on a visually distinct subset rather than one that generalises.

The honest position: the frequency signal's discrimination on this set is not
explained by resampling, but cannot yet be attributed to synthesis artefacts
rather than subject matter. Reweighting on this evidence would optimise the
system for FFHQ faces against digital art. The weights are left unchanged and
the question is put on the roadmap, where it needs a set with varied content at
matched native resolution to settle.

---

## 9. Track B — the provenance track

16 files, 13 labelled (4 generated, 9 real), 3 unlabelled plumbing files
excluded from metrics.

**Counts, not percentages.** With 13 labelled files a single reclassification
moves "accuracy" by roughly eight points, so ratios here convey precision the
sample size does not support.

| outcome | count |
|---|---|
| decided | 8 |
| abstained (UNCERTAIN) | 5 |
| true positive (AI called AI) | 2 |
| false negative (AI called REAL) | 1 |
| false positive (REAL called AI) | 2 |
| true negative (REAL called REAL) | 3 |

### Per-signal availability

This is the point of the track. Metadata was available on **9 of 16** files, and
all three of its outcomes occurred on genuine files:

| outcome | files | example detail string |
|---|---|---|
| generator fingerprint, score 1.0 | 4 | `C2PA / Content Credentials manifest found (raw scan)` — Adobe Firefly render |
| | | `Generator signature found: openai (raw scan)` — ChatGPT image |
| camera EXIF, score 0.0 | 4 | `Camera EXIF present (Make/Model), no generator signature` — iPhone 15 Pro |
| unavailable | 5 | WhatsApp images and video — EXIF stripped in transit |

The camera-EXIF branch had never once fired before these files existed; a third
of the provenance detector was untested until now. The WhatsApp files
demonstrate the stripping behaviour the design anticipates: the platform removes
EXIF, the signal correctly reports *unavailable* rather than a low score, and
the remaining weights renormalise.

### The ensemble caught a classifier failure

The classifier is confidently wrong on real iPhone photographs. On `IMG_5019.jpg`
— a genuine camera original — it returned **0.998**.

| signal | score | nominal weight |
|---|---|---|
| classifier | 0.998 | 0.60 |
| provenance (camera EXIF) | 0.000 | 0.25 |
| frequency | 0.091 | 0.15 |

Fused: `(0.998×0.60 + 0.000×0.25 + 0.091×0.15) / 1.00` = **0.613** →
UNCERTAIN.

Had provenance been unavailable, the same file would have fused to
`(0.998×0.60 + 0.091×0.15) / 0.75` = **0.817** → AI_GENERATED, a confident false
positive on a real photograph.

**The provenance signal converted a false positive into an abstention on three
of the four camera originals.** This is the ensemble thesis demonstrated rather
than asserted: an independent signal pulled back an over-confident classifier.

It also shows the limit. The WhatsApp images had their EXIF stripped, received
no such correction, and produced two false positives — including one at
classifier score 0.994. **When the platform strips provenance, the ensemble
loses the signal that would have saved it.** That is the honest characterisation
of where this system is weak in the field.

## 10. Generator specialisation (D8)

All ten worst misclassifications by margin, from the normalised run, are in the
same direction: **AI called REAL**. The system's failures are misses, not false
alarms — consistent with precision 0.8112 against recall 0.6170.

Of 50 generators, **6 were never caught once** and 8 were caught every time.
The split is not random:

| caught every time | mean classifier score |
|---|---|
| `bguisard/stable-diffusion-nano-2-1` | 0.994 |
| `cgburgos/sdxl-1-0-base` | 0.966 |
| `IDK-ab0ut/Yiffymix_v36` | 0.999 |
| `Masagin/Deliberate` | 0.941 |

| never caught | mean classifier score |
|---|---|
| `aliyualisa/model` | 0.000 |
| `KORguy/textual_inversion_shirt` | 0.008 |
| `kalebanana/textual_inversion_mvtec` | 0.026 |
| `VegaKH/Ultraskin` | 0.034 |
| `naclbit/trinart_stable_diffusion_v2` | 0.059 |
| `briannlongzhao/2` | 0.060 |

The detector is `Organika/sdxl-detector`. It recognises SDXL and its close
relatives almost perfectly, and misses small community fine-tunes, textual
inversions and merges — several of them SD 1.5-era rather than SDXL. This is a
**generalisation boundary with a mechanism behind it**, not an unexplained error
rate: the classifier detects what it was trained to detect, and a generated
image from outside that family passes as real with near-zero confidence.

Caveats: all 200 generated images are of a single architecture family
(`LatDiff`), so no architecture-level comparison is possible; and with 4 images
per generator, individual slip rates are coarse.

## 11. Limitations and future work

**Measured and unresolved**

- Weights are unfitted, and §8 shows the ensemble underperforming its best
  component. Resolving it needs an evaluation set with varied content at matched
  native resolution — the current set cannot distinguish a synthesis signal from
  a subject-matter one.
- A better threshold band (0.06 / 0.68, held-out accuracy 0.8492) exists and was
  declined; §4.3 gives the reasoning. Revisit once a less confounded set exists.
- The classifier's generalisation boundary (§10) suggests the model signal
  should be treated as a *detector of SDXL-family output*, not of
  machine-generated media generally.

**Implementation gaps**

- Real C2PA manifest parsing is not implemented; detection is a bounded raw-byte
  scan for markers plus EXIF text tags. The `c2pa` package requires a native
  build. Consequently the system reports that a manifest is *present*, never
  who signed it or whether the claim chain validates.
- Video container metadata uses the same bounded raw scan rather than proper
  MP4 atom parsing.
- Clips longer than 60 seconds are analysed over their first 60 seconds only;
  the response says so. Spreading the same 60-frame budget across the full
  duration would be a better use of it.
- `.heic` is not an accepted upload format. The four camera originals in
  `samples/` were converted to JPEG with EXIF preserved for evaluation; adding
  HEIC support would change the frozen contract's accepted-extensions list.

**Evaluation gaps**

- Track B has 13 labelled files. It demonstrates that each provenance branch
  works; it does not measure how often they work.
- No video appears in Track A, so the frame-sampling and aggregation path is
  exercised only by Track B's single clip and the plumbing fixtures.
- The frequency signal's constants bound the range where the ratio moves and
  encode no measured AI-versus-real boundary. Its score means "how
  high-frequency is this image", not "is this image AI".
