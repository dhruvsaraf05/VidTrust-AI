import axios from 'axios'

import imageResponse from '../mocks/image_response.json'
import videoResponse from '../mocks/video_response.json'
import { mediaTypeOf } from '../lib/validation'

export const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8000'

export const MODES = { MOCK: 'mock', LIVE: 'live' }

/**
 * Analyse a file.
 *
 * Always resolves -- it never throws. Callers get one of:
 *   { ok: true,  data }
 *   { ok: false, error: CODE, message }
 *
 * IMPORTANT (backend/API_NOTES.md): the contract error code lives in the
 * response BODY, not the HTTP status. Statuses are 415/413/422/500/503, and
 * branching on them would miss cases and couple the UI to numbers the backend
 * never froze. So we read `error` off the body and ignore the status entirely.
 *
 * NETWORK_ERROR is the one code that does NOT come from the backend. It means
 * the request never produced a response body at all (server down, CORS, DNS).
 * It is a client-side condition and is deliberately distinct from the five
 * contract codes.
 *
 * There is no request timeout. A 60-second video takes 15-20s on CPU, and a
 * fixed timeout would abort legitimate work. Progress is reported through
 * onProgress instead.
 */
export async function analyze(file, { mode = MODES.LIVE, onProgress } = {}) {
  if (mode === MODES.MOCK) {
    return mockAnalyze(file, onProgress)
  }

  const form = new FormData()
  form.append('file', file)

  try {
    const response = await axios.post(`${API_BASE}/api/analyze`, form, {
      timeout: 0,
      onUploadProgress: (event) => {
        if (!onProgress || !event.total) return
        onProgress({
          phase: 'uploading',
          percent: Math.round((event.loaded / event.total) * 100),
        })
      },
    })
    onProgress?.({ phase: 'done', percent: 100 })
    return { ok: true, data: response.data }
  } catch (error) {
    const body = error.response?.data

    if (body && typeof body.error === 'string') {
      return { ok: false, error: body.error, message: body.message ?? '' }
    }

    return {
      ok: false,
      error: 'NETWORK_ERROR',
      message: `Could not reach the backend at ${API_BASE}. Is it running?`,
    }
  }
}

/** Serves the captured fixtures so the UI can be built with no server. */
async function mockAnalyze(file, onProgress) {
  const type = mediaTypeOf(file.name)
  const fixture = type === 'video' ? videoResponse : imageResponse

  // Roughly the real timings so the progress state gets exercised honestly.
  const duration = type === 'video' ? 2600 : 700
  const started = performance.now()

  await new Promise((resolve) => {
    const tick = setInterval(() => {
      const elapsed = performance.now() - started
      const percent = Math.min(100, Math.round((elapsed / duration) * 100))
      onProgress?.({ phase: percent < 100 ? 'uploading' : 'analyzing', percent })
      if (elapsed >= duration) {
        clearInterval(tick)
        resolve()
      }
    }, 80)
  })

  onProgress?.({ phase: 'done', percent: 100 })

  // Reflect the real filename back so the fixture doesn't look mismatched.
  return { ok: true, data: { ...fixture, filename: file.name } }
}

/**
 * Backend health. Used for the classifier-down indicator.
 *
 * Returns the body as-is on success. `status` is "ok" or "degraded"; when
 * degraded, `error` is "MODEL_UNAVAILABLE" and the API is still usable -- the
 * metadata and frequency signals keep answering. That is the ensemble design
 * working as intended, not an outage.
 */
export async function health() {
  try {
    const response = await axios.get(`${API_BASE}/api/health`, { timeout: 5000 })
    return { ok: true, data: response.data }
  } catch {
    return { ok: false, error: 'NETWORK_ERROR' }
  }
}
