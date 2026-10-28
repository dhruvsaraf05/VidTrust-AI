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
| Classifier | `model` | 0.60 | `haywoodsloan/ai-image-detector-deploy` (SwinV2), inference only, no fine-tuning; fallback `Ateeqq/ai-vs-human-image-detector` |
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

### The classifier was replaced on 10 September — by measurement

The original classifier, `Organika/sdxl-detector`, was not merely weak on
real-world files but **anti-correlated**: it scored genuine iPhone originals in
`samples/` at 0.998 "artificial" and a Gemini-generated image at 0.003. No
weight or threshold can repair a signal pointing the wrong way, so the model was
swapped rather than the fusion tuned around it.

The replacement was chosen by `quick_compare.py`, which scores every cached
candidate over the same 12 labelled sample images (4 generated,
8 real), inference only:

| candidate | correct at 0.5 | mean score, generated | mean score, real | generated caught |
|---|---|---|---|---|
| `haywoodsloan/ai-image-detector-deploy` | 12/12 | 0.987 | 0.002 | 4/4 |
| `Ateeqq/ai-vs-human-image-detector` | 11/12 | 0.752 | 0.001 | 3/4 |
| `dima806/ai_vs_human_generated_image_detection` | 8/12 | 0.012 | 0.011 | 0/4 |
| `Organika/sdxl-detector` | 5/12 | 0.438 | 0.574 | 2/4 |
| `dima806/ai_vs_real_image_detection` | 4/12 | 0.766 | 0.694 | 4/4 |

`haywoodsloan/ai-image-detector-deploy` is a SwinV2 (744 MB) with the label
vocabulary `artificial` / `real`; the fallback's labels are `ai` / `hum`. Both
are used exactly as published. **Every Track A figure below was re-measured on
the replacement**; nothing from the original classifier's runs is carried
forward except where it is explicitly labelled as history.

One consequence must be stated now rather than discovered later: the twelve
images that chose the model are twelve of Track B's thirteen labelled files.
Track B is therefore the model-selection set as well as the demo set, and §9
treats it accordingly.

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

This was not a formality. With the original classifier, a threshold search
over all 400 rows at once suggested a single cut near 0.06 lifting accuracy to
roughly 0.73; under a proper split on the same data the selection half chose a
completely different cut, 0.806, which reached only 0.6850 on the held-out
half. The apparent gain was an artefact of choosing and reporting on the same
data. (Those two figures describe the replaced model and are kept only as the
methodological lesson; the current model's split is in §4.3.)

## 4. Track A results

### 4.1 Headline

| | naive | **normalised** |
|---|---|---|
| accuracy | 0.7595 | **0.8384** |
| precision | 0.7107 | **0.8152** |
| recall | 0.8731 | 0.8731 |
| F1 | 0.7836 | **0.8431** |
| ROC-AUC | 0.9196 | **0.9487** |
| coverage | 0.9875 | 0.9900 |
| abstained | 5 | 4 |

Confusion matrix, normalised condition, positive class = AI, 4 abstentions
excluded:

| | predicted AI | predicted REAL |
|---|---|---|
| **actual AI** | 172 | 25 |
| **actual REAL** | 39 | 160 |

Signal availability across all 400: model 400, **metadata 0**, frequency 400.
Every run is therefore "degraded" by definition.

Recall is identical in both conditions — 172 caught, 25 missed — and that is
not a coincidence. Generated images in this set are natively 512×512, so the
512 centre crop leaves them untouched (the per-image classifier and frequency
scores are byte-identical across the two CSVs for all 200). Only the real
images, cropped from 1024², change. **The entire normalised gain is 31 fewer
false positives on real photographs**, which is exactly what removing the
resampling confound should do if the confound was inflating the real side.

The error profile is the mirror image of the original classifier's: precision
now trails recall. The system's characteristic mistake is calling a real FFHQ
portrait generated (70 naive, 39 normalised), not letting a generated image
through.

### 4.2 Ablation

Recomputed from the per-signal scores dumped by `evaluate.py`; no inference is
re-run. Each configuration re-fuses using the same renormalisation rule the
live pipeline uses, so dropping a signal is exactly equivalent to that signal
being unavailable at runtime.

| configuration | naive acc | naive AUC | norm acc | norm AUC |
|---|---|---|---|---|
| full ensemble | 0.7595 | **0.9196** | 0.8384 | **0.9487** |
| model only | 0.7563 | 0.8397 | 0.8367 | 0.8947 |
| metadata only | — | 0.5000 | — | 0.5000 |
| frequency only | 0.7194 | 0.9012 | 0.7177 | 0.9129 |
| ensemble − metadata | 0.7595 | 0.9196 | 0.8384 | 0.9487 |
| ensemble − frequency | 0.7563 | 0.8397 | 0.8367 | 0.8947 |

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
| naive | single cut 0.808 | 0.8350 | 0.8650 | 1.00 |
| naive | band 0.14 / 0.82 | 0.9220 | 0.9444 | 0.72 |
| naive | current 0.35 / 0.65 | — | 0.7437 | 0.995 |
| normalised | single cut 0.808 | 0.8700 | 0.8800 | 1.00 |
| normalised | band 0.10 / 0.82 | 0.9586 | **0.9664** | 0.745 |
| normalised | current 0.35 / 0.65 | — | 0.8342 | 0.995 |

**Neither measured operating point was adopted.** `config.py` still holds
0.35 / 0.65.

Two things are visible in this table and both are stated rather than
smoothed over. First, the band result is real and survives the split: 0.9664
against 0.8342 on data never used to choose it. It was declined because it
costs **25 points of coverage** — abstaining on a quarter of inputs instead of
one in two hundred — and a detector that declines to answer on a quarter of what
it is shown is a different product, not a tuned one.

Second, and new with this classifier: a plain single cut at 0.808 with **no
abstention band at all** reaches 0.8650 / 0.8800 held-out at full coverage,
against 0.7437 / 0.8342 for the current band. That is a coverage-free gain, and
it says plainly that the hand-chosen 0.65 threshold sits too low for this
classifier's score distribution: the real portraits it is unsure about fuse into
the 0.65–0.80 range and are called generated. It was still not adopted, for the
same reason as before — the cut is fitted to a score distribution shaped by the
dataset confounds in §5, on a set of face photographs versus digital art, which
is not what the demonstration inputs look like. On the 13 hand-collected files
the current band makes no error (§9), so there is no field evidence yet that
the threshold is wrong, only evidence that it is wrong on FFHQ.

Both findings are recorded; the constants are not changed. Moving them would be
tuning to hide a limitation instead of reporting it.

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

## 8. Named finding (b): the ensemble now out-ranks its best component — it did not before

With the replacement classifier the fused ensemble reaches **AUC 0.9196 naive
and 0.9487 normalised**, against 0.9012 / 0.9129 for frequency alone and
0.8397 / 0.8947 for the classifier alone. The ensemble beats every one of its
constituents in both conditions.

**This is a reversal, and it should be read as one.** With the original
classifier the same table read 0.7918 / 0.8461 for the ensemble against
0.9012 / 0.9129 for frequency alone: the fused system was worse than its
cheapest signal, the one carrying the smallest weight. That finding was
reported at the time rather than tuned away, and the numbers are retained here
as history so the reversal is visible. (They describe the replaced model; the
current ones are in §4.2.)

**What changed and what did not.** The frequency signal is the same code on the
same files and reports the same AUC in both eras — 0.9012 and 0.9129, to four
places. The fusion rule and the weights are unchanged. Only the classifier
moved. A classifier that ranks poorly on its own dragged a 0.60-weighted average
below the frequency signal; one that ranks well combines with it to something
better than either. The ensemble did not become good because it was tuned. It
became good because its heaviest input stopped being wrong.

**The caveat on frequency stands unchanged.** Its discrimination on this set is
not explained by resampling — under normalisation it rose rather than fell —
but it still cannot be attributed to synthesis artefacts rather than subject
matter. A 512×512 centre crop of a 1024×1024 FFHQ portrait is mostly smooth
skin, and the generated images are whole detailed scenes; smooth-versus-detailed
is precisely what an FFT ratio measures. Frequency-only precision of 1.000 at
recall 0.324 is consistent with a signal that fires confidently on a visually
distinct subset. The ensemble's margin over the classifier alone (+0.080 naive,
+0.054 normalised AUC) arrives through the frequency signal, so that margin
inherits the same caveat.

**The weights remain unfitted.** 0.60 / 0.25 / 0.15 were chosen by inspection
before any measurement existed, and the fact that they now produce a better
result than any single signal is not evidence that they are right — it is
evidence that the classifier is now good enough for a prior of roughly this
shape to help rather than hurt. Fitting them is still on the roadmap and still
needs a set with varied content at matched native resolution.

---

## 9. Track B — the provenance track

16 files, 13 labelled (4 generated, 9 real), 3 unlabelled plumbing files
excluded from metrics.

**Counts, not percentages.** With 13 labelled files a single reclassification
moves "accuracy" by roughly eight points, so ratios here convey precision the
sample size does not support.

| outcome | count |
|---|---|
| decided | 13 |
| abstained (UNCERTAIN) | 0 |
| true positive (AI called AI) | 4 |
| false negative (AI called REAL) | 0 |
| false positive (REAL called AI) | 0 |
| true negative (REAL called REAL) | 9 |

**A perfect column is expected here and is not evidence.** Twelve of these
thirteen files are the images that `quick_compare.py` used to choose the
classifier (§2); the thirteenth is the WhatsApp video. A model selected for
scoring 12/12 on a set will score 12/12 on it. This track was never the
accuracy measurement — Track A is — and after the model swap it is doubly
disqualified from being one. What it still measures is provenance, below.

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

### What the model swap changed here

With the original classifier this track held the report's clearest
demonstration of the ensemble thesis. `IMG_5019.jpg`, a genuine iPhone
original, was scored **0.998** by that classifier; camera EXIF at 0.000 and
frequency at 0.091 pulled the fusion down to 0.613 — UNCERTAIN — where without
provenance it would have fused to 0.817 and been called generated. Provenance
converted a false positive into an abstention on three of the four camera
originals, and the two WhatsApp images, with EXIF stripped in transit, got no
such correction and were called generated at classifier scores up to 0.994.

That behaviour is precisely why the classifier was replaced (§2). On the
replacement the same file scores **0.006**, every camera original and every
WhatsApp image fuses below 0.10, and there is no over-confident classifier left
for provenance to correct. The demonstration no longer occurs on these files.
The figures above are retained as history because they are the reason the
model changed; they are reproducible by pointing `MODEL_PRIMARY` back at the
original and re-running `evaluate.py --track samples`.

The mechanism is unchanged and is still the design's answer to the field case
this track exposed: **when a platform strips provenance, the ensemble loses the
signal that would have saved it.** The WhatsApp files still show the stripping
— metadata unavailable, weights renormalised to 0.80 / 0.20 — and are now
decided correctly by the classifier alone. That is a better classifier
covering for a missing signal, not the missing signal being any less missing.

## 10. Generator specialisation (D8)

Of the ten worst misclassifications by margin in the normalised run, nine are
**AI called REAL** and one is **REAL called AI** — an FFHQ portrait at
classifier score 0.998, fused 0.889. The misses are shallow: the deepest sits at
fused confidence 0.040, a margin of 0.31 below the REAL threshold, and the
tenth-worst at 0.157.

Of 50 generators, **none was missed on every image**, 29 were caught on all
four, and the worst slip rate is 2 of 4 on seven generators:

| slipped 2 of 4 | mean classifier score |
|---|---|
| `danbochman/ccxl` | 0.547 |
| `AACEE/textual_inversion_sksship` | 0.514 |
| `aliyualisa/model` | 0.457 |
| `nota-ai/bk-sdm-small` | 0.543 |
| `selshiya/teapot` | 0.514 |
| `bguisard/stable-diffusion-nano-2-1` | 0.646 |
| `WarriorMama777/AbyssOrangeMix` | 0.571 |

**The error structure has changed shape, not just size.** With the original
classifier, misses clustered by generator family — six generators were never
caught, eight always were, and the split had a mechanism: an SDXL detector
recognising SDXL-family output and passing SD 1.5-era fine-tunes and textual
inversions as real at near-zero confidence. With the replacement, misses are
thin and spread. No generator is systematically missed; the worst-slipping ones
carry mean classifier scores of 0.46–0.65 rather than near zero, and three of
them (`aliyualisa/model`, `nota-ai/bk-sdm-small`,
`bguisard/stable-diffusion-nano-2-1`) were in the original model's
never-caught list. That is the signature of a general-purpose detector
operating near its margin on some images, not of a specialist outside its
training family.

The cost moved to the other side. The classifier's false alarms on real
portraits — 39 of 200 normalised, 70 naive — now dominate the error count and
are the reason precision (0.8152) trails recall (0.8731). Every real image in
this set is an FFHQ face, so this is a measurement of one real-image source
only; how the classifier behaves on real photographs of anything else is not
measured here. The four iPhone originals and four WhatsApp photographs in Track
B, all scored below 0.01, are the only non-FFHQ real images it has been run on.

Caveats: all 200 generated images are of a single architecture family
(`LatDiff`), so no architecture-level comparison is possible; and with 4 images
per generator, individual slip rates are coarse — one image is 25 points.

## 11. Limitations and future work

**Measured and unresolved**

- Weights are unfitted. §8 shows them producing a better result than any
  single signal with this classifier, which is not the same as showing they are
  right. Fitting them needs an evaluation set with varied content at matched
  native resolution — the current set cannot distinguish a synthesis signal from
  a subject-matter one.
- Two better operating points exist and were declined: a band (0.10 / 0.82,
  held-out accuracy 0.9664 at coverage 0.745) and a plain single cut (0.808,
  held-out 0.8800 at full coverage). §4.3 gives the reasoning; the second in
  particular says the current 0.65 threshold is too low for this classifier on
  FFHQ. Revisit once a less confounded set exists.
- The classifier's dominant error is false alarms on real FFHQ portraits (§10),
  measured on one real-image source. Its behaviour on real photographs of other
  subjects rests on eight Track B files.
- Track B is now the model-selection set as well as the demo set (§2, §9). A
  fresh hand-collected set, not used to choose anything, is needed before any
  hand-collected figure can be read as an estimate.
- Inference cost. The replacement is a SwinV2 with 744 MB of weights. Measured
  through the API on the demo laptop (4-core i5, CPU, threads capped at half):
  **~4–6 s per image**, 18.8 s for a 6-frame clip, 46.6 s for a 13-frame clip.
  A 60-frame clip extrapolates to roughly 3.5–4 minutes and has not been timed.
  The original model answered an image in under half a second; the frontend's
  progress state, not a timeout, is what makes the new cost survivable.

**Implementation gaps**

- Real C2PA manifest parsing is not implemented; detection is a bounded raw-byte
  scan for markers plus EXIF text tags. The `c2pa` package requires a native
  build. Consequently the system reports that a manifest is *present*, never
  who signed it or whether the claim chain validates.
- Video container metadata uses the same bounded raw scan rather than proper
  MP4 atom parsing.
- Long clips are sampled across their whole duration: the 60-frame budget is
  spread evenly, so a 10-minute video is read one frame every 10 s rather than
  over its first minute only. Short clips sample exactly as before (a 13 s clip
  still gives 13 frames, covered by a test). The cost is per sampled frame, so
  the URL duration cap rose from 60 s to 10 minutes. A 60-frame clip has still
  not been timed end to end.
- **URL ingest downloads third-party platform media.** YouTube's Terms of
  Service restrict downloading, and other platforms have comparable terms. It
  is used here for academic research on short public clips, retaining nothing —
  media is deleted in a `finally` block on every path. Instagram serves nothing
  to an anonymous client (a made-up post ID and a real one fail identically),
  so the link path now uses **the operator's own signed-in browser session, for
  Instagram, Facebook and Threads only**; every other host is fetched
  anonymously, so an arbitrary URL is never handed the cookie jar. That is
  automated access which Instagram's terms restrict, and a session cookie is a
  credential: it belongs on a local demo machine, never a shared server.
  Setting `URL_BROWSER_COOKIES = None` in `config.py` turns it off. Whether
  reels now fetch reliably end to end has not been measured in this report.
- **Provenance is structurally unavailable for every URL.** Platforms re-encode
  on upload, discarding EXIF/XMP/C2PA, so the metadata signal renormalises out
  on every URL analysis. The interface says so explicitly rather than reporting
  a generic absence — the same effect measured on the WhatsApp files in Track B
  (§9), now visible at the point of use.
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

## 12. The operating envelope: robustness to real-world degradation

Every figure above is on images as distributed. Real media is recompressed,
resized and pushed through messaging apps, so this section measures what
happens then. `perturbations.py` defines eleven degradations;
`run_robustness_sweep.py` applies each to all 400 Track A images **after** the
§6 centre crop, so both classes enter every condition with identical processing
history (order: crop, perturb, save losslessly; recorded in each metrics JSON).
`robustness.py` reads the per-condition CSVs and writes `robustness.json`. It
runs no inference.

Provenance is unavailable on all 400 files throughout, so these are two-signal
figures. "Flagged" is the number of generated images labelled `AI_GENERATED`
at the deployed 0.35/0.65 band, out of 200.

| Condition | Ensemble AUC | Classifier AUC | Frequency AUC | Generated flagged | Real flagged |
|---|---|---|---|---|---|
| clean (crop only) | 0.9487 | 0.8947 | 0.9129 | 172 | 39 |
| JPEG q95 | 0.9097 | 0.8177 | 0.9128 | 149 | 40 |
| JPEG q75 | 0.9314 | 0.7950 | 0.9132 | 129 | 19 |
| JPEG q50 | 0.9328 | 0.7845 | 0.9119 | 110 | 14 |
| JPEG q30 | 0.9355 | 0.7683 | 0.9098 | **95** | 5 |
| downscale ×0.75 | 0.9480 | 0.8956 | 0.9068 | 176 | 46 |
| downscale ×0.50 | 0.9306 | 0.8681 | 0.9016 | 179 | 71 |
| crop to 256² | 0.9157 | 0.8488 | 0.9117 | 150 | 37 |
| noise σ=2 | 0.9219 | 0.8437 | 0.9117 | 182 | 97 |
| noise σ=5 | 0.9107 | 0.8159 | 0.9046 | 132 | 25 |
| noise σ=10 | 0.9344 | 0.7446 | 0.8818 | **75** | 0 |
| messaging app | 0.9623 | 0.8588 | 0.9068 | 122 | 5 |


### 12.1 AUC is nearly blind to the failure that matters

From clean to JPEG q30 the ensemble AUC moves 0.9487 → 0.9355, a change of
0.013 that reads as "robust". Over the same range the generated images actually
flagged fall **172 → 95 of 200, a 45% loss of detections**. The per-signal dump
shows why: the mean classifier score on generated images falls 0.862 → 0.485,
below the 0.65 threshold, while on real images it falls further, 0.225 →
0.046. The *ordering* survives, which is all AUC measures, but the generated
distribution slides out of `AI_GENERATED` into `UNCERTAIN`. Precision rises,
0.8152 → 0.9500: the degraded system is not making more mistakes, it is
declining to answer.

Mild recompression is not an attack; it is what every messaging app does. The
messaging composite (long edge to 1080 in **either** direction, then JPEG 70)
keeps AUC at 0.9623, higher than clean, while flagging 122 of 200. Under noise
σ=10 only 75 remain.

### 12.1a Light degradation fails the other way: false alarms

Reading detections alone, rescaling and light noise look harmless, even
helpful: 176 and 179 flagged after downscaling, 182 after noise σ=2, all
above clean's 172. The real-image column says otherwise. These conditions push
**real** images up the classifier's scale:

| Condition | Mean classifier score, real | Real flagged | Accuracy |
|---|---|---|---|
| clean | 0.225 | 39 | 0.8384 |
| downscale ×0.50 | 0.393 | 71 | 0.7679 |
| noise σ=2 | **0.511** | **97** | 0.7096 |
| noise σ=5 | 0.155 | 25 | 0.7683 |

So the failure has two directions. Light resampling or sensor-like noise makes
real photographs read as generated (at σ=2, nearly half of the 200 real
images are flagged). Heavier compression or noise makes generated images read
as real. Noise is not monotone: σ=2 lifts the real-image mean, while σ=5 and
σ=10 lower both classes. The earlier draft of this analysis, and the first
draft of the paper, said rescaling "costs nothing"; that was read off the
detections column and is wrong.

### 12.2 Without the control, the conclusion inverts

Three conditions were also run on the uncontrolled set, at native resolutions.
Classifier ΔAUC against the matching clean baseline:

| Condition | With the crop control | Without |
|---|---|---|
| JPEG q50 | −0.110 | **+0.029** |
| messaging | −0.036 | **+0.004** |
| noise σ=10 | −0.150 | −0.300 |

Two of three change sign. A protocol that degraded images as they come would
report that compression and platform re-encoding *improve* this detector. The
mechanism is §6 again: noise on a 1024² real image is partly averaged away by
the internal resize to 512, while on a native 512² generated image it is not,
so the degradation itself carries the label.

### 12.3 The frequency signal is compression-invariant, which points to content

Across all four JPEG levels the frequency AUC spans 0.9098–0.9132, against
0.9129 clean. The prediction was the opposite. If the signal read the spectral
trace of generative up-sampling, JPEG — which quantises exactly those
high-frequency coefficients — should erase it. A statistic that survives
quantisation of the band it claims to read is not reading that band's
artefacts. The economical explanation is gross image structure: smooth aligned
FFHQ portraits against detailed generated illustration. This strengthens §5's
confound and the §8 caution. The signal stays labelled unfitted; if a
content-matched set confirms this, setting its weight to zero is the correct
outcome.

### 12.4 Where this reaches the user

The Reliability Card (`reliability.py`, `POST /api/reliability`, and a
`reliability` block on `/api/analyze-url`) uses these figures. It reads the
input's condition with no model (format, estimated JPEG quality from the
quantisation tables, surviving EXIF, whether a platform delivered it), maps it
to the nearest measured condition, and shows the generated and real images
flagged there next to clean originals. It leads with detections, not AUC, for
the reason in §12.1, and states the caveats: still images only, an upper bound
given §5, two-signal, unfitted thresholds.

## 13. The confidence is not a probability

`calibration.py` measures whether the fused confidence can be read as P(AI).
It cannot.

- **ECE 0.1154** on the held-out half (0.1054 over all 400).
- **The empirical rate is not monotone in confidence.** Over all 400 files, the
  0.7–0.8 bin holds 32 files of which 31.2% are generated; the 0.1–0.2 bin
  holds 28 of which 46.4% are.
- **Platt scaling makes it worse**, fitted on the selection half and scored on
  the held-out half: ECE 0.1363 and Brier 0.1241, against 0.1154 and 0.1187
  raw. That is expected when the rate is non-monotone rather than merely
  mis-scaled, and no monotone recalibration can fix it.

So the confidence is a **ranking score**. The interface shows it as a position
on a scale, not as "87% probability", and should keep doing so.

**Fitted fusion, reported and not adopted.** Logistic regression over the two
available per-signal scores, under leave-one-generator-out cross-validation
(50 folds), reaches out-of-fold AUC **0.9674** against the hand-chosen
weights' 0.9487. The implied relative weights are 0.26 classifier and
0.74 frequency, i.e. a strong shift toward the signal §12.3 argues is
measuring content. Adopting it would fit the confound, the same reasoning by
which §4.3 declined the better thresholds. `calibration.json` records it with
`"adopted": false`.

## 14. Explaining a verdict, and a heatmap we declined to draw

`POST /api/explain` returns the evidence behind each signal for one image,
shown in the interface under "Show evidence":

- **Frequency:** the log-magnitude spectrum the score is computed from, with
  the low-frequency disc outlined. It is rendered by the same function that
  computes the score, and a test asserts the two agree.
- **Provenance:** every metadata field the scan read (EXIF, XMP, embedded
  text), which one matched a generator signature, and the raw-scan result.
  The collector was restructured so the drill-down and the signal read the
  same fields; a test asserts every `samples/` file still gives exactly its
  committed provenance reading.
- **Classifier:** a Grad-CAM heatmap, **not drawn**, for the reason below.

### 14.1 Measured before shown

`saliency_faithfulness.py` compares the stage-2 Grad-CAM against occlusion
(blank each 32×32 cell of the 256×256 input and record the drop in the
'artificial' logit) on 20 evaluation images, 10 per class, centre-cropped.
Agreement is Spearman ρ over the 64 cells.

| | All 20 | Generated | Real |
|---|---|---|---|
| median ρ | **−0.02** | −0.05 | +0.03 |
| range | −0.47 to +0.76 | −0.47 to +0.76 | −0.22 to +0.34 |
| ρ > 0 | 9 | 3 | 6 |
| flatness gate would draw | 16 | 9 | 7 |
| … of which ρ > 0 | 8 | 3 | 5 |

The heatmap is no better than chance at locating what the classifier responds
to. The flatness gate that was meant to stop a meaningless map does not help:
it rejects *flat* maps, not *wrong* ones, and the maps it lets through agree
half the time. So `explain()` now has a faithfulness gate first: below a
measured median of ρ = 0.3 (PROVISIONAL, a judgement of "clearly better than
chance", not fitted), no map is drawn and the response says why. That also
skips the gradient pass, so the panel is faster. A map that looks like an
explanation without being one would be the most persuasive overclaim this
project could make.

### 14.2 A claim this corrected

An earlier note, from one image, said the classifier's evidence was "spatially
diffuse", because the worst single patch cost about 10% of the logit margin.
Twenty images do not support that. On 3 of 10 generated images, one cell moves
the logit by 1.5 to 3.7. (The first run also reported a "share of the logit"
that exceeded 5 and went negative; the raw 'artificial' logit is often near
zero, so that ratio was meaningless and was replaced by the raw drop.) Whether
a large single-cell drop is evidence, or the model reacting to an unnatural
grey square, occlusion cannot settle. What the study does settle is that
Grad-CAM does not find those cells.
