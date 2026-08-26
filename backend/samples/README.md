# Sample files

Drop test media here, then record what each file actually is in `MANIFEST.CSV`.

Run the sweep with the server already up:

```
.venv/Scripts/python run_samples.py
```

## What to collect

The point of this set is **not** to make the detector look good. It is to find
out where it breaks, before the mentor does. A set of 20 easy AI images would
tell you nothing and would fall apart under the first hard question.

Aim for roughly this spread:

### Real, metadata intact (4 files)
Straight off your phone, transferred by cable or Google Drive **as original
files**. These are the only files that exercise the "genuine camera EXIF"
branch of the metadata signal, so without them one third of that detector is
untested.

### Real, metadata stripped (3 files)
The same kind of photo, but sent through WhatsApp or posted to Instagram and
downloaded back. Every social platform strips EXIF. This is the *common* real-
world case and it must report `metadata.available: false` — not a low score.

### AI, metadata intact (4 files)
Downloaded directly from the generator: Midjourney, DALL-E via ChatGPT, Adobe
Firefly, Google Imagen, SDXL via a local run or DreamStudio — whatever you
actually have access to. **Save the original download. Do not screenshot.**
A screenshot destroys every byte of provenance metadata and silently converts
an easy case into a hard one.

### AI, metadata stripped (3 files)
AI images that have been screenshotted or re-saved. These are the honest hard
cases: the metadata signal goes unavailable and the classifier has to carry the
verdict alone. Expect these to be your weakest results, and say so in the demo
rather than hoping nobody asks.

### Video (4 files)
Two real clips off your phone, two AI clips if you can get them (Sora, Veo,
Runway, Pika). If you cannot get AI video tonight, that is a finding to state
plainly, not a gap to paper over — the video path is still demonstrably working
on real clips.

### Deliberate breakage (2 files + the ones below)
- one `.webp` and one `.avi` or `.mov`, to prove the less common extensions work
- a file over 50 MB → must return `FILE_TOO_LARGE`
- a `.txt` renamed to `.png` → must return `PROCESSING_FAILED`
- a `.pdf` or `.gif` → must return `UNSUPPORTED_FORMAT`

## Two warnings

**Do not relabel a file to match the output.** If the detector calls your
holiday photo `AI_GENERATED`, that is a result. Record it. A false positive you
can explain is worth more in a viva than a clean table you cannot.

**Screenshots are not the original.** If half your "AI" samples are screenshots,
the metadata signal will be unavailable across the board and you will conclude
it does not work, when in fact it was never given anything to read.
