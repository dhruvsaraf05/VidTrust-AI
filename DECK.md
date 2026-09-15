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
| Classifier | 0.60 | `haywoodsloan/ai-image-detector-deploy`, inference only |
| Provenance | 0.25 | EXIF / XMP / C2PA fingerprints |
| Frequency | 0.15 | FFT high-frequency energy ratio |

`≥ 0.65` AI_GENERATED · `≤ 0.35` LIKELY_REAL · else UNCERTAIN

> No training, no fine-tuning — the classifier is used as published. The
> signals are independent by design: a neural judgement, a document-metadata
> lookup, and a spectral measurement. They can disagree, and when they do that
> disagreement is visible rather than averaged away.
>
> If asked which classifier: it was swapped on 10 September. The original,
> an SDXL-specific detector, scored genuine iPhone photos 0.998 "artificial"
> — anti-correlated, not merely weak. Four candidates were scored on the same
> labelled files by `quick_compare.py`; this one was 12/12. Every number on
> the following slides was re-measured on it.

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
| accuracy | 0.7595 | **0.8384** |
| precision | 0.7107 | **0.8152** |
| recall | 0.8731 | 0.8731 |
| ROC-AUC | 0.9196 | **0.9487** |

| configuration | naive AUC | norm AUC |
|---|---|---|
| full ensemble | **0.9196** | **0.9487** |
| model only | 0.8397 | 0.8947 |
| metadata only | 0.5000 | 0.5000 |
| frequency only | 0.9012 | 0.9129 |

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
> — see the next slide's note if asked why crop rather than resize. Recall is
> identical in both conditions because the generated images are natively 512
> and the crop does not touch them; the whole gain is 31 fewer false alarms on
> real photographs, which is what removing a resampling confound should do.

---

## 6 — Two findings

### (a) A three-character string inverted the provenance signal

`"veo"` matched compressed binary by chance and reported
**"generator found", score 1.0, on 62 of 200 real photographs.**

It fired more often on real images than generated ones. Invisible until
measured.

### (b) The ensemble now out-ranks every single signal — it did not before

**full ensemble AUC 0.9196 / 0.9487 · classifier alone 0.8397 / 0.8947 ·
frequency alone 0.9012 / 0.9129**

With the original classifier the ensemble was *worse* than frequency alone.
Same weights, same fusion, same frequency code — only the classifier changed.

We report these results as they are — no manual tuning of weights to hide the
limitations.

> On (a): every unit-level behaviour was correct — the scan found the string it
> was told to find. The API returned well-formed responses. Only a labelled set
> plus a per-signal dump could surface it. This is the argument for why
> evaluation is not optional, and it is the strongest thing in the project.
>
> On (b): say the reversal plainly. With the first classifier we reported that
> the fused system was worse than its cheapest signal, and we did not touch the
> weights to fix that. The classifier was replaced because it was anti-
> correlated on real photographs, and with a classifier that ranks well on its
> own the same unfitted weights now combine to something better than any
> single signal. The weights did not become right; the heaviest input stopped
> being wrong.
>
> The frequency caveat is unchanged. The obvious explanation for its strength
> was resolution — real images are 1024², generated are 512² — so we
> controlled for it, and frequency-only did not collapse; it rose. But a
> content confound remains: every real image is an FFHQ face, and a 512 crop
> of a face is mostly smooth skin while the generated images are detailed
> scenes. Smooth-versus-detailed is precisely what an FFT ratio measures. The
> ensemble's margin over the classifier alone comes through frequency, so it
> inherits that caveat.

---

## 7 — Limitations

**Dataset (all three inflate the numbers):** zero EXIF · every real image is
FFHQ · real 1024² vs generated 512², no overlap

**Measured and declined:** band 0.10 / 0.82 reaches held-out accuracy
**0.9664** vs 0.8342 but abstains on a quarter of inputs; a plain cut at 0.808
reaches 0.8800 at full coverage. Neither adopted.

**Where it fails now:** false alarms on real FFHQ portraits (39 of 200
normalised) dominate. No generator is missed on every image; 29 of 50 are
caught on all four.

**Cost:** the replacement classifier is a 744 MB SwinV2 — ~4–6 s per image on
CPU, ~47 s for a 13-frame clip.

> The declined thresholds are worth defending as a decision, not a gap. Both
> survive a proper selection/reporting split, so they are real. The band
> abstains on 26% of inputs instead of under 1%. The single cut is the more
> uncomfortable one: it says our hand-chosen 0.65 sits too low for this
> classifier on FFHQ, at no coverage cost. We still left it, because it is
> tuned to a confounded distribution of faces versus art, and on the
> hand-collected files the current band makes no error. Tuning it would be
> hiding a limitation rather than reporting one.
>
> The generator picture changed shape with the model. The old classifier was an
> SDXL specialist — six generators never caught, with a mechanism. The new one
> misses thinly and everywhere: worst case 2 of 4, no generator systematically.
> Its cost moved to false alarms on real portraits, and that is measured on one
> real-image source only — FFHQ.
>
> If the demo feels slow, say why: the new classifier is ten times heavier than
> the old one, and the interface shows progress rather than pretending.

---

## 8 — Roadmap

1. **Resolution- and content-matched evaluation set** — the prerequisite for
   settling finding (b) and for any reweighting
2. **Fit the weights** against that set, then revisit the threshold band
3. **Real C2PA parsing** — report the signing authority, not merely that a
   manifest exists
4. **Video at scale** — Track A has no video; the frame path is barely measured
5. **A fresh hand-collected set** that was not used to choose the classifier —
   the current `samples/` chose it, so its perfect score is not evidence
6. **Second classifier or a lighter one** — the SwinV2's per-image cost limits
   video; ensemble a second model and re-run the same evaluation before
   trusting either

> The honest close: this system does not claim to detect all machine-generated
> media, does not generalise to unseen generators without re-evaluation, and
> does not measure content quality. What it measures is provenance — was this
> machine-generated — and the evaluation says precisely how well, and where it
> stops working.
>
> Saying that before being asked is worth more than any single accuracy figure.
