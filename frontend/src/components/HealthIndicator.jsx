import { MODES } from '../api/client'

/**
 * Backend and classifier status. Fetched once by App and passed in.
 *
 * The middle state is the interesting one: the backend can be up while the
 * classifier is down, and /api/analyze still answers using the metadata and
 * frequency signals with their weights renormalised. That is the ensemble
 * design working, so it reads as a caution rather than an outage.
 *
 * MODEL_UNAVAILABLE only ever arrives here, never from /api/analyze.
 *
 * A glyph accompanies every state — the dot's colour is never the only thing
 * distinguishing them.
 */
export default function HealthIndicator({ mode, state }) {
  const view = describe(mode, state)

  return (
    <span className="flex items-center gap-2" title={view.title}>
      <span
        aria-hidden="true"
        className="h-1.5 w-1.5 rounded-full"
        style={{ backgroundColor: view.dot }}
      />
      <span className="label" style={{ color: view.text }}>
        {view.label}
      </span>
    </span>
  )
}

function describe(mode, state) {
  if (mode === MODES.MOCK) {
    return {
      label: 'Fixture data',
      dot: 'var(--color-ink-3)',
      text: 'var(--color-ink-3)',
      title: 'Serving captured responses from src/mocks — the backend is not being called',
    }
  }

  if (!state) {
    return {
      label: 'Checking',
      dot: 'var(--color-rule-2)',
      text: 'var(--color-ink-3)',
      title: 'Contacting the backend',
    }
  }

  if (!state.ok) {
    return {
      label: 'Backend offline',
      dot: 'var(--color-crimson)',
      text: 'var(--color-crimson)',
      title: 'No response from the backend',
    }
  }

  if (state.data.status === 'degraded') {
    return {
      label: 'Classifier down · 2 of 3 signals',
      dot: 'var(--color-ochre)',
      text: 'var(--color-ochre)',
      title: `${state.data.error}: ${state.data.model_error ?? 'the classifier did not load'}`,
    }
  }

  return {
    label: 'All 3 signals ready',
    dot: 'var(--color-teal)',
    text: 'var(--color-ink-2)',
    title: state.data.model_name ?? '',
  }
}
