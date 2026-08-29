import { useEffect, useState } from 'react'

import { MODES, health } from '../api/client'

/**
 * Backend / classifier status.
 *
 * Three states worth distinguishing, and the middle one is the interesting one:
 *
 *   ok         backend up, classifier loaded
 *   degraded   backend up, classifier DOWN -- /api/analyze still works, using
 *              the metadata and frequency signals alone. This is the ensemble
 *              design doing its job, so it reads as a caution, not an outage.
 *   offline    no response at all
 *
 * `MODEL_UNAVAILABLE` only ever arrives here, never from /api/analyze.
 */
export default function HealthIndicator({ mode }) {
  const [state, setState] = useState(null)

  useEffect(() => {
    // Mock mode renders its own branch below, so no fetch and no reset.
    if (mode === MODES.MOCK) return
    let cancelled = false
    health().then((result) => {
      if (!cancelled) setState(result)
    })
    return () => {
      cancelled = true
    }
  }, [mode])

  if (mode === MODES.MOCK) {
    return (
      <span className="flex items-center gap-2">
        <span className="h-1.5 w-1.5 rounded-full bg-ink-faint" />
        <span className="micro">Fixture data</span>
      </span>
    )
  }

  if (!state) {
    return (
      <span className="flex items-center gap-2">
        <span className="h-1.5 w-1.5 rounded-full bg-rule-strong" />
        <span className="micro">Checking…</span>
      </span>
    )
  }

  if (!state.ok) {
    return (
      <span className="flex items-center gap-2" title="No response from the backend">
        <span className="h-1.5 w-1.5 rounded-full bg-ai" />
        <span className="micro text-ai">Backend offline</span>
      </span>
    )
  }

  const degraded = state.data.status === 'degraded'

  return (
    <span
      className="flex items-center gap-2"
      title={
        degraded
          ? `${state.data.error}: ${state.data.model_error ?? 'classifier did not load'}`
          : state.data.model_name ?? ''
      }
    >
      <span
        className={`h-1.5 w-1.5 rounded-full ${degraded ? 'bg-uncertain' : 'bg-real'}`}
      />
      <span className={`micro ${degraded ? 'text-uncertain' : ''}`}>
        {degraded ? 'Classifier down — 2 of 3 signals' : 'Backend ready'}
      </span>
    </span>
  )
}
