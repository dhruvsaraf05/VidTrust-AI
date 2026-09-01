# VidTrust AI — machine-generated media detector

Final-year B.Tech (Information Technology) capstone, Somaiya Vidyavihar
University. Semester VII.

Detects **machine-generated media** — fully synthetic images and video from
generators such as SDXL, Midjourney, DALL·E, Firefly and their community
fine-tunes. It is not a face-swap detector; see [Scope](#scope).

Three independent signals are fused by weighted average, and a signal that
cannot run is removed from the average rather than scored as zero.

| Signal | Weight | What it measures |
|---|---|---|
| Classifier | 0.60 | `Organika/sdxl-detector` AI-probability |
| Provenance | 0.25 | EXIF / XMP / C2PA generator fingerprints |
| Frequency | 0.15 | FFT high-frequency energy ratio |

Fused confidence → verdict: `>= 0.65` AI_GENERATED, `<= 0.35` LIKELY_REAL,
otherwise UNCERTAIN.

---

## Scope

Semester VI was a deepfake **face-swap** detector (XceptionNet on
FaceForensics++). That project is complete and is not extended here. Nothing in
this pipeline inspects faces — there is no face detector and no facial crop
step. The name is retained; the scope is not.

No figure from the Semester VI report appears anywhere in this one.

---

## Requirements

- **Python 3.14** (verified on 3.14.4; torch ships a `cp314` win_amd64 wheel)
- **Node 22+** for the frontend
- Windows paths are used below (`.venv/Scripts/`); on Linux/macOS substitute
  `.venv/bin/`

---

## Backend

Torch must be installed from the CPU index **first**, or pip will try to
resolve CUDA builds:

```bash
cd backend
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install -r requirements.txt
```

Run the API:

```bash
cd backend && .venv/Scripts/python -m uvicorn main:app --reload
```

Serves on `http://127.0.0.1:8000`. The classifier is downloaded on first start
(~5 min once, cached afterwards) and loaded a single time at startup, never per
request. If it fails to load the API still answers, using the other two
signals — `GET /api/health` reports `status: degraded`.

Verify:

```bash
curl -s http://127.0.0.1:8000/api/health
```

```bash
cd backend && curl -s -X POST http://127.0.0.1:8000/api/analyze -F "file=@samples/_smoke_tagged.jpg"
```

## Frontend

```bash
cd frontend && npm install && npm run dev
```

Open **http://localhost:5173** — use `localhost`, not `127.0.0.1`; Vite binds
IPv6 and the backend's CORS allows the `localhost` origin.

The **Mock / Live** toggle in the header switches between captured fixtures in
`src/mocks/` and the real API, so the UI can be developed with the backend
stopped. Fonts are vendored in `public/fonts` — no CDN, so a lecture-hall
network cannot break the demo.

---

## Reproducing every number

Every figure in `REPORT.md` comes from one of these commands. None is
hand-computed.

**1. Build the evaluation set** (~400 images, Community Forensics, CVPR 2025).
It is not committed — images are gitignored — so a fresh clone must rebuild it.
Expect ~1.1 GB of downloads and 15–20 minutes; the HuggingFace datasets-server
rate-limits sustained fetching:

```bash
cd backend && .venv/Scripts/python build_evaluation_set.py --real 200 --fake 200
```

**2. Evaluate**, both conditions:

```bash
cd backend && .venv/Scripts/python evaluate.py --track public
```

```bash
cd backend && .venv/Scripts/python evaluate.py --track public --normalise crop
```

```bash
cd backend && .venv/Scripts/python evaluate.py --track samples
```

**3. Ablation and threshold selection** (recomputed from the per-signal dump,
no re-inference):

```bash
cd backend && .venv/Scripts/python ablation.py --report evaluation_report_public.csv --compare evaluation_report_public_normalised.csv
```

**4. Failure analysis:**

```bash
cd backend && .venv/Scripts/python failure_analysis.py --report evaluation_report_public_normalised.csv
```

**5. Frequency calibration and the sample sweep:**

```bash
cd backend && .venv/Scripts/python calibrate_frequency.py samples/
```

```bash
cd backend && .venv/Scripts/python run_samples.py
```

Outputs (`evaluation_report_*.csv`, `evaluation_metrics_*.json`,
`ablation.json`, `failure_analysis.json`) are gitignored: they are regenerated,
not authored.

---

## Layout

```
backend/
├── main.py                     FastAPI app, routes, error handlers, lifespan
├── config.py                   every tunable constant; single source of truth
├── detectors/                  model / metadata / frequency
├── pipeline/                   video_sampler, aggregator (fusion)
├── build_evaluation_set.py     D4 — materialise the labelled set
├── evaluate.py                 D5 — metrics, two tracks, optional normalisation
├── ablation.py                 D6/D7 — ablation table, threshold selection
├── failure_analysis.py         D8 — worst errors, per-generator attribution
├── run_samples.py              sweep samples/, record behaviour
├── calibrate_frequency.py      measure the FFT ratio spread
├── API_NOTES.md                contract deviations and integration notes
└── samples/                    hand-collected set + MANIFEST.csv ground truth
frontend/                       React + Vite + Tailwind
REPORT.md                       evaluation write-up
CLAUDE.md                       working agreements and invariants
PRD_DEMO_SEPT_3.md              milestone definition
```

---

## API

`POST /api/analyze` — `multipart/form-data`, single field `file`.
Accepts `.jpg .jpeg .png .webp .mp4 .mov .avi`, max 50 MB.

Errors return `{"error": CODE, "message": "..."}`. **Branch on the `error`
field, not the HTTP status.** Full contract, including the two codes that are
deliberately unreachable, is in [backend/API_NOTES.md](backend/API_NOTES.md).

`GET /api/health` reports classifier status, weights, thresholds and limits.
The frontend reads its verdict thresholds from here rather than hardcoding
them.

---

## Known limitations

Stated in full in [REPORT.md](REPORT.md). In brief:

- Weights (0.60/0.25/0.15) and thresholds (0.65/0.35) are **hand-chosen and
  unfitted**. A better-performing threshold band was measured and deliberately
  not adopted — see the report.
- The public evaluation set has **zero EXIF**, all real images come from
  **FFHQ**, and the classes differ in **native resolution**. All three inflate
  results; the normalised condition controls the third.
- Real C2PA manifest parsing is not implemented — detection is a bounded
  raw-byte scan for markers plus EXIF text tags.
- Clips longer than 60 seconds are analysed over their first 60 seconds.
