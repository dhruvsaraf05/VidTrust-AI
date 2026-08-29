# CLAUDE.md

Ground truth for any Claude Code session working in this repository.
If something here contradicts what you infer from the code, **say so rather
than silently picking one** — a stale line in this file is a bug.

---

## 1. What this project is

**VidTrust AI (Semester VII)** — a detector for **machine-generated media**:
fully synthetic images and video from generators like Midjourney, SDXL,
DALL·E, Firefly, Sora and Veo.

Final-year B.Tech (Information Technology) capstone, Somaiya Vidyavihar
University. Two-semester project; this is the second semester.

### What this project is NOT

**This is not a deepfake face-swap detector.** Semester VI was — it used
XceptionNet trained on FaceForensics++, and it was submitted and evaluated.
That work is finished and is not being extended.

Nothing in the Semester VII pipeline inspects faces. There is no face
detector, no MTCNN, no facial crop step, no XceptionNet. If a task seems to
call for one, the task has been misread — stop and ask.

The name "VidTrust AI" is retained from Semester VI and still appears in
`main.py` and docstrings. The scope did not carry over with the name.

**Do not cite accuracy figures from the Semester VI report** (94.7%, AUC 0.978
or any others) anywhere — code, comments, report, slides. Nothing goes in the
Semester VII report that isn't reproducible from a script in this repo.

---

## 2. Team and working agreements

One active contributor: **Dhruv** (backend, ML, and now frontend). A second
member was to build the frontend but became unavailable; assume no other help.

- **Dhruv handles all git himself.** Do not run `git` commands — no staging,
  committing, branching, pushing or merging. When a stage is working, say it's
  a good commit point and suggest a message. He runs it.
- **No model training or fine-tuning.** The classifier is used pretrained, as
  is. Semester constraints make training out of scope.
- Prefer the smallest thing that works. If an approach costs more than about
  20 minutes, offer the cheaper version first and mark a `TODO`.
- Never fabricate sample data, evaluation numbers, or accuracy figures. If
  measurement hasn't happened, the answer is "not measured yet."
- Constants that were chosen by inspection must be labelled `PROVISIONAL` or
  `UNFITTED` in the code, and described that way in prose.

Branches: `main` (integration), `feat/backend`, `feat/frontend`.

---

## 3. Environment

Windows. Python **3.14.4** — this is fine; torch ships a cp314 win_amd64
wheel. Do not suggest downgrading.

```
cd backend
.venv/Scripts/python -m uvicorn main:app --reload    # start server
.venv/Scripts/python run_samples.py                  # sweep samples/
.venv/Scripts/python calibrate_frequency.py samples/ # FFT calibration
```

Interpreter path is `.venv/Scripts/python`, not `.venv/bin/python`. Torch must
be installed from the CPU index first:

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

`transformers` resolves to 5.x. Use `device="cpu"`, not `device=-1`.
Server runs on `http://127.0.0.1:8000`. Frontend dev server on `:5173`.

---

## 4. Layout

```
backend/
├── main.py                    # FastAPI app, routes, error handlers, lifespan
├── config.py                  # every tunable constant; single source of truth
├── detectors/
│   ├── model_detector.py      # HF classifier, loaded once at startup
│   ├── metadata_detector.py   # EXIF / XMP / raw C2PA marker scan
│   └── frequency_detector.py  # FFT high-frequency energy ratio
├── pipeline/
│   ├── video_sampler.py       # frame sampling; raises VideoReadError
│   └── aggregator.py          # analyze_image / analyze_video, fusion
├── run_samples.py             # sweeps samples/, writes samples_report.{json,csv}
├── calibrate_frequency.py     # measures FFT ratio spread across a folder
├── requirements.txt
└── samples/
    ├── README.md              # what to collect and why
    ├── MANIFEST.csv           # ground truth: AI | REAL | UNKNOWN
    └── _smoke_*               # plumbing tests only — never demo material
frontend/                      # React + Vite + Tailwind
API_NOTES.md                   # contract deviations and integration notes
```

---

## 5. Architecture

Three **independent** signals fused by weighted average. The independence is
the point of the design, not an implementation detail.

| Signal | Key | Weight | What it measures |
|---|---|---|---|
| Classifier | `model` | 0.60 | `Organika/sdxl-detector` AI-probability |
| Provenance | `metadata` | 0.25 | EXIF/XMP/C2PA generator fingerprints |
| Frequency | `frequency` | 0.15 | FFT high-frequency energy ratio |

Fused confidence → verdict: `>= 0.65` → `AI_GENERATED`, `<= 0.35` →
`LIKELY_REAL`, otherwise `UNCERTAIN`.

Video is images over time: sample ~1 fps capped at 60 frames, run the
classifier per frame, metadata once on the container, frequency on every 5th
frame, then aggregate.

**Weights and thresholds are hand-chosen and unfitted.** Deriving them from a
labelled evaluation set is the current top-priority work item (see the PRD).

---

## 6. Invariants — do not break these

These encode decisions that were made deliberately, several after finding a
bug. Changing any of them requires asking first.

1. **An unavailable signal is renormalised out of the average, never scored
   as 0.0.** A missing signal must not drag a verdict toward "real". This is
   the single most important behaviour in the fusion logic.
2. **Never map generic classifier labels** (`LABEL_0`, `LABEL_1`). Without a
   model card you cannot know which index means "AI", and a wrong guess
   silently inverts the whole signal. Unrecognised vocabulary must set
   `available: false`.
3. **The classifier loads once, in the lifespan handler.** Never on the
   request path.
4. **Every error leaves through the contract shape** `{"error": CODE,
   "message": ...}` — including FastAPI validation errors and unhandled
   exceptions. No `{"detail": [...]}` envelopes escape.
5. **Never leak server filesystem paths** into a client-facing message.
6. **A dead classifier is not a 500.** It degrades to `available: false` and
   the other two signals still answer. `MODEL_UNAVAILABLE` is reported only on
   `GET /api/health`.
7. **`NO_FACES_DETECTED` is unreachable** and stays that way. It exists only
   so the frozen contract is honoured. Do not add a code path that raises it.
8. **`frames[].score` is classifier-only**, not a fused per-frame confidence.
   Do not plot it without checking `signals.model.available`.
9. **All signals unavailable → confidence `0.5`, verdict `UNCERTAIN`.**
   Total signal failure must not read as `LIKELY_REAL`.
10. **Never evaluate on files used for tuning or demo.** The evaluation set and
    `samples/` stay separate.
11. **Never relabel a sample to match the detector's output.** A false positive
    that gets recorded and explained is worth more than a clean table.

---

## 7. The API contract

`POST /api/analyze` — `multipart/form-data`, single field `file`.

The 200 body is **frozen**. Field names, types, signal keys, verdict strings
and the `frames` shape must not change. If a change looks necessary, stop and
tell Dhruv first.

```json
{
  "id": "a3f9c1",
  "filename": "sample_04.png",
  "media_type": "image",
  "verdict": "AI_GENERATED",
  "confidence": 0.87,
  "processing_time_ms": 1420,
  "signals": {
    "model":     { "name": "Classifier", "score": 0.91, "weight": 0.6,
                   "detail": "sdxl-detector", "available": true },
    "metadata":  { "name": "Provenance", "score": 1.0, "weight": 0.25,
                   "detail": "C2PA manifest found: Midjourney v6", "available": true },
    "frequency": { "name": "Frequency analysis", "score": 0.64, "weight": 0.15,
                   "detail": "Elevated high-frequency energy", "available": true }
  },
  "frames": null
}
```

`frames` is `null` for images; for video an array of
`{ "index", "timestamp", "score" }`.

Error codes and statuses:

| code | status | reachable? |
|---|---|---|
| `UNSUPPORTED_FORMAT` | 415 | yes — bad extension, empty file, malformed multipart |
| `FILE_TOO_LARGE` | 413 | yes — over 50 MB, enforced while streaming |
| `NO_FACES_DETECTED` | 422 | **no** — Semester VI leftover |
| `PROCESSING_FAILED` | 500 | yes — corrupt media, unhandled exception |
| `MODEL_UNAVAILABLE` | 503 | **not on `/api/analyze`** |

The frontend must branch on the `error` field, **not** on `res.status`.

`GET /api/health` is additive: model status, configured weights and
thresholds, size limit, accepted extensions.

Timing to design UI around: images ~300–450 ms, a 6-second video ~1.8 s, a
60-second video 15–20 s on CPU. The upload UI needs a progress state, not a
fixed timeout.

---

## 8. Known limitations — state these, don't hide them

- **The frequency signal is unfitted.** `FFT_RATIO_MIN/MAX` (0.05/0.45) bound
  the range where the ratio *moves*; they encode no measured AI-vs-real
  boundary. Right now the score means "how high-frequency is this image", not
  "is this AI". Present it that way until calibration says otherwise. If
  evaluation shows it doesn't separate the classes, **setting its weight to
  zero and reporting that is the correct outcome**, not a failure.
- Real C2PA manifest parsing is not implemented — the `c2pa` package needs a
  native build. Current detection is a bounded raw-byte scan for markers plus
  EXIF text tags.
- Video container metadata uses a bounded raw scan, not proper MP4 atom
  parsing.
- Clips longer than 60 seconds are analysed over the first 60 seconds only.
  The response says so.
- Weights and thresholds are hand-chosen.
- `samples/_smoke_*` files prove the pipeline runs. They say nothing about
  accuracy and must never appear in a demo or report.

---

## 9. Current state

Backend stages 1–5 complete and verified on `feat/backend`: images, video, all
three detectors, all reachable error paths, sample harness.

Outstanding: real samples collected and labelled, frequency calibration run,
the evaluation set and `evaluate.py`, the ablation study, and the frontend.

See `PRD_DEMO_SEPT_3.md` for what those mean and when they're due.
