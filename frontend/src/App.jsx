import { useEffect, useState } from 'react'

import { MODES, analyze } from './api/client'
import HealthIndicator from './components/HealthIndicator'
import ModeToggle from './components/ModeToggle'
import UploadZone from './components/UploadZone'
import { styleForVerdict } from './lib/verdict'

const MODE_KEY = 'vidtrust.mode'

export default function App() {
  const [mode, setMode] = useState(
    () => localStorage.getItem(MODE_KEY) ?? MODES.MOCK,
  )
  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState(null)
  const [result, setResult] = useState(null)
  const [failure, setFailure] = useState(null)

  useEffect(() => {
    localStorage.setItem(MODE_KEY, mode)
  }, [mode])

  async function handleAnalyze(file) {
    setBusy(true)
    setProgress(null)
    setResult(null)
    setFailure(null)

    const outcome = await analyze(file, { mode, onProgress: setProgress })

    if (outcome.ok) {
      setResult(outcome.data)
    } else {
      // Branch on the body's `error` field, never on the HTTP status.
      setFailure({ code: outcome.error, message: outcome.message })
    }
    setBusy(false)
  }

  return (
    <div className="min-h-screen">
      <header className="border-b border-rule bg-surface">
        <div className="mx-auto flex max-w-3xl items-center justify-between gap-6 px-6 py-4">
          <div>
            <h1 className="text-sm font-semibold tracking-[0.14em] text-ink uppercase">
              VidTrust AI
            </h1>
            <p className="micro mt-1 normal-case tracking-normal">
              Machine-generated media detector
            </p>
          </div>
          <div className="flex items-center gap-5">
            <HealthIndicator mode={mode} />
            <ModeToggle mode={mode} onChange={setMode} disabled={busy} />
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-3xl px-6 py-10">
        <UploadZone onAnalyze={handleAnalyze} busy={busy} progress={progress} />

        {failure && (
          <div className="mt-6 border-l-2 border-ai bg-ai-soft px-4 py-3">
            <p className="num text-xs tracking-wide text-ai">{failure.code}</p>
            <p className="mt-1 text-sm text-ink">{failure.message}</p>
          </div>
        )}

        {result && <ResultPlaceholder result={result} />}
      </main>
    </div>
  )
}

/**
 * SCAFFOLD ONLY.
 *
 * VerdictCard, SignalPanel and FrameTimeline are Sunday's work (PRD D3). This
 * panel exists so the response can be seen flowing end to end tonight, and it
 * should be deleted when the real components land -- it is not a design.
 *
 * The one thing it does do properly is render `available: false` as the word
 * "unavailable" rather than a 0% bar, because a missing signal is renormalised
 * out of the average and is NOT evidence of authenticity.
 */
function ResultPlaceholder({ result }) {
  const style = styleForVerdict(result.verdict)

  return (
    <section className="mt-8">
      <div className={`flex items-baseline gap-3 border ${style.border} ${style.bg} px-4 py-3`}>
        <span className={`h-2 w-2 rounded-full ${style.dot}`} />
        <span className={`text-sm font-semibold ${style.text}`}>{style.label}</span>
        <span className="num ml-auto text-sm text-ink">
          {result.confidence.toFixed(3)}
        </span>
      </div>

      <dl className="mt-4 divide-y divide-rule border border-rule bg-surface">
        {Object.entries(result.signals).map(([key, signal]) => (
          <div key={key} className="flex items-baseline gap-4 px-4 py-3">
            <dt className="w-40 shrink-0">
              <span className="text-sm text-ink">{signal.name}</span>
              <span className="micro ml-2 num">w={signal.weight}</span>
            </dt>
            <dd className="min-w-0 flex-1 text-sm text-ink-muted">
              {signal.detail}
            </dd>
            <dd className="num shrink-0 text-sm">
              {signal.available ? (
                signal.score.toFixed(3)
              ) : (
                <span className="micro">unavailable</span>
              )}
            </dd>
          </div>
        ))}
      </dl>

      <p className="micro mt-3">
        {result.media_type} · {result.frames ? `${result.frames.length} frames` : 'no frames'} ·{' '}
        <span className="num">{result.processing_time_ms}</span> ms · id{' '}
        <span className="num">{result.id}</span>
      </p>

      <details className="mt-4">
        <summary className="micro cursor-pointer select-none hover:text-ink">
          Raw response
        </summary>
        <pre className="num mt-2 max-h-80 overflow-auto border border-rule bg-surface-sunken p-3 text-[11px] leading-relaxed text-ink-muted">
          {JSON.stringify(result, null, 2)}
        </pre>
      </details>
    </section>
  )
}
