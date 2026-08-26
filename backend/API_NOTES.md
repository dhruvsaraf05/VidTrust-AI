# Backend API notes — for the frontend integration

Server: `http://127.0.0.1:8000`. CORS allows `http://localhost:5173` and
`http://127.0.0.1:5173`.

The 200 response matches the frozen contract exactly. Field names, types,
signal keys, verdict strings and `frames` shape are all as specified.

## Open questions for the frontend — these need an answer before hour 6

The contract froze the **response body**. It did not specify these, and I had
to pick something to get the endpoint running. Each one is a place where the
two branches could disagree silently.

1. **HTTP status codes on errors.** The contract fixed the error *body*
   (`{"error": CODE, "message": "..."}`) but not the status. Current mapping:

   | code | status |
   |---|---|
   | `UNSUPPORTED_FORMAT` | 415 |
   | `FILE_TOO_LARGE` | 413 |
   | `NO_FACES_DETECTED` | 422 |
   | `PROCESSING_FAILED` | 500 |
   | `MODEL_UNAVAILABLE` | 503 |

   **Branch on the `error` field, not on the status code.** If the frontend is
   currently keying off `res.status === 400`, it will miss every one of these.

2. **`NO_FACES_DETECTED` is unreachable.** It is a leftover from the Semester VI
   face-swap scope. Nothing in the Semester VII pipeline looks at faces, so no
   code path raises it. It stays defined so the contract is honoured. If the
   frontend has a UI state for it, that state is dead — worth knowing before it
   ends up in the demo script or the slides.

3. **`MODEL_UNAVAILABLE` is never returned by `/api/analyze`.** By design: if
   the classifier fails to load, that signal reports `available: false` and the
   other two still produce a verdict. Returning 503 instead would throw away a
   working answer. The classifier's status is exposed on `GET /api/health`.

4. **`frames[].score` is the classifier score for that frame** — not a fused
   per-frame confidence. Metadata runs once on the container and frequency runs
   on every 5th frame, so neither has a per-frame value. If the classifier is
   unavailable, frame scores are `0.0` and `signals.model.available` is `false`;
   don't plot frame scores without checking that flag.

5. **Malformed multipart** (missing or misnamed `file` field) returns
   `UNSUPPORTED_FORMAT` / 415, not FastAPI's default
   `{"detail": [...]}` envelope. Everything leaves through the contract shape.

6. **If every signal is unavailable**, confidence is `0.5` and the verdict is
   `UNCERTAIN`. That is consistent with the frozen thresholds (0.35 < 0.5 <
   0.65) and avoids a total failure reading as `LIKELY_REAL`.

## Additive, not a contract change

`GET /api/health` returns model load status, the configured weights and
thresholds, the size limit and the accepted extensions. Useful for a "backend
connected" indicator and for confirming the classifier actually loaded.

## Things the frontend should expect

- `processing_time_ms` covers the whole request including upload spooling.
  Images are ~300–450 ms; a 6-second video is ~1.8 s. A 60-second video will be
  around 15–20 s on CPU — the upload UI needs a spinner, not a fixed timeout.
- `signals.*.detail` is free text meant to be shown to the user. It is not a
  stable enum; do not parse it.
- `available: false` should render as "unavailable", never as a 0% bar. That
  distinction is the point of the design.
