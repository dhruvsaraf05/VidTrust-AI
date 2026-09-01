# Demo 2 — slide deck

Eight slides. Every figure traces to a script; see
[README.md](README.md#reproducing-every-number). Speaker notes are the
indented blocks — say these, don't put them on the slide.

---

## 1 — The scope changed, and why

**Semester VI:** face-swap detection. XceptionNet, FaceForensics++. Submitted,
defended, closed.

**Semester VII:** machine-generated media. Nothing here inspects faces.

> The threat model moved. A swapped face assumes there *was* a photograph to
> alter. The media that now needs checking was never captured by any camera —
> there is no original, and no face to compare against. Detecting a swap is
> irrelevant to an image that is synthetic end to end.
>
> Volunteer up front: the name is inherited, the scope is not. And no number
> from last semester's report appears anywhere in this one.

---

## 2 — Architecture

Three independent signals → weighted average → verdict.

| Signal | Weight | Evidence |
|---|---|---|
| Classifier | 0.60 | `Organika/sdxl-detector`, inference only |
| Provenance | 0.25 | EXIF / XMP / C2PA fingerprints |
| Frequency | 0.15 | FFT high-frequency energy ratio |

`≥ 0.65` AI_GENERATED · `≤ 0.35` LIKELY_REAL · else UNCERTAIN

> No training, no fine-tuning — the classifier is used as published. The
> signals are independent by design: a neural judgement, a document-metadata
> lookup, and a spectral measurement. They can disagree, and when they do that
> disagreement is visible rather than averaged away.

---

## 3 — Why an ensemble: the renormalisation rule

```
confidence = Σ(score × weight) / Σ(weight)     over AVAILABLE signals
```

**A missing signal is removed from the average. It is never scored 0.0.**

Metadata absent → classifier and frequency rescale from 0.60/0.15 to
**0.80/0.20**.

> This is the design decision the whole project rests on, so give it a moment.
>
> 0.0 means "this looks real". Absence means "no evidence either way". If you
> conflate them, every file whose metadata was stripped drifts toward a verdict
> of authentic — and stripping metadata is what every social platform does on
> upload. So the missing signal is renormalised out, and the interface draws it
> as a hatched "no reading" track, never an empty bar.
>
> If nothing is available at all, confidence is held at 0.500, not 0.

---

## 4 — Live demo

1. Real photograph, EXIF intact → provenance reads the camera
2. AI image, no metadata → carried by the classifier
3. **Adobe Firefly render → provenance fires independently**
4. Video → per-frame timeline
5. A deliberate failure, explained before anyone finds it

> Step 3 is the moment worth pausing on: the C2PA manifest is read straight out
> of the file, with no model involved at all. Point at the ledger — the weight
> column shows `.25 ▸ .25` when provenance is present and `.25 ▸ —` when it is
> not, with the other rows visibly widening to absorb it.
>
> Step 5 is not an accident. Volunteering the failure is worth more than hoping
> nobody asks.

---

## 5 — Track A results and the ablation

**400 images, Community Forensics (CVPR 2025), 50 generators, 200/200 balanced.**

| | naive | normalised |
|---|---|---|
| accuracy | 0.6532 | **0.7302** |
| precision | 0.6705 | **0.8112** |
| recall | 0.6170 | 0.6170 |
| ROC-AUC | 0.7918 | **0.8461** |

| configuration | naive AUC | norm AUC |
|---|---|---|
| full ensemble | 0.7918 | 0.8461 |
| model only | 0.7064 | 0.7482 |
| metadata only | 0.5000 | 0.5000 |
| frequency only | **0.9012** | **0.9129** |

> Two things to say before anyone asks.
>
> Metadata is available on 0 of 400 — public datasets are redistributed
> re-encoded, which strips EXIF. So the six-row ablation has only three
> distinct results: "ensemble minus metadata" *is* the full ensemble, and
> "ensemble minus frequency" *is* model-only. Those are arithmetic identities
> forced by the dataset, not independent findings. That is exactly why there is
> a second track.
>
> "Normalised" means every image was centre-cropped to 512×512 of native pixels
> — see the next slide's note if asked why crop rather than resize.

---

## 6 — Two findings

### (a) A three-character string inverted the provenance signal

`"veo"` matched compressed binary by chance and reported
**"generator found", score 1.0, on 62 of 200 real photographs.**

It fired more often on real images than generated ones. Invisible until
measured.

### (b) The ensemble is worse than its best component

**frequency-only AUC 0.9129 · full ensemble 0.8461**

Weights are hand-chosen and unfitted. Reweighting would fit a confound.

> On (a): every unit-level behaviour was correct — the scan found the string it
> was told to find. The API returned well-formed responses. Only a labelled set
> plus a per-signal dump could surface it. This is the argument for why
> evaluation is not optional, and it is the strongest thing in the project.
>
> On (b): say it plainly, don't soften it. Then explain why we did *not* act on
> it. The obvious explanation was resolution — real images are 1024², generated
> are 512² — so we controlled for it, and frequency-only did not collapse; it
> rose. That is a real finding. But a content confound remains: every real image
> is an FFHQ face, and a 512 crop of a face is mostly smooth skin while the
> generated images are detailed scenes. Smooth-versus-detailed is precisely what
> an FFT ratio measures. Reweighting on that would optimise for FFHQ faces
> against digital art.

---

## 7 — Limitations

**Dataset (all three inflate the numbers):** zero EXIF · every real image is
FFHQ · real 1024² vs generated 512², no overlap

**Measured and declined:** threshold band 0.06 / 0.68 reaches held-out accuracy
**0.8492** vs 0.6885 — costs 29 points of coverage. Not adopted.

**Generalisation boundary:** 6 of 50 generators never caught once. All ten worst
errors are *AI called REAL*.

> The declined threshold is worth defending as a decision, not a gap. It
> survives a proper selection/reporting split, so it is real — but it abstains
> on 37% of inputs instead of 8%, and it is tuned to a confounded distribution
> of faces versus art, which is not what the demo files look like.
>
> The generator finding has a mechanism: the model is an *SDXL* detector. It
> catches `sdxl-1-0-base` at 0.966 and misses small community fine-tunes and
> textual inversions at near-zero. It detects what it was trained to detect.

---

## 8 — Roadmap

1. **Resolution- and content-matched evaluation set** — the prerequisite for
   settling finding (b) and for any reweighting
2. **Fit the weights** against that set, then revisit the threshold band
3. **Real C2PA parsing** — report the signing authority, not merely that a
   manifest exists
4. **Video at scale** — Track A has no video; the frame path is barely measured
5. **Broaden the classifier** beyond the SDXL family, or ensemble a second one

> The honest close: this system does not claim to detect all machine-generated
> media, does not generalise to unseen generators without re-evaluation, and
> does not measure content quality. What it measures is provenance — was this
> machine-generated — and the evaluation says precisely how well, and where it
> stops working.
>
> Saying that before being asked is worth more than any single accuracy figure.
