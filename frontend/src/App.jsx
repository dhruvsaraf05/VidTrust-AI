import { useCallback, useEffect, useState } from 'react'

import { MODES, analyze, health } from './api/client'
import ErrorState from './components/ErrorState'
import FrameTimeline from './components/FrameTimeline'
import HealthIndicator from './components/HealthIndicator'
import ModeToggle from './components/ModeToggle'
import RawResponse from './components/RawResponse'
import SignalLedger from './components/SignalLedger'
import ThresholdTrack from './components/ThresholdTrack'
import UploadZone from './components/UploadZone'
import VerdictBlock from './components/VerdictBlock'
import { DEFAULT_THRESHOLDS } from './lib/verdict'

const MODE_KEY = 'vidtrust.mode'

export default function App() {
  const [mode, setMode] = useState(
    () => localStorage.getItem(MODE_KEY) ?? MODES.MOCK,
  )
  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState(null)
  const [result, setResult] = useState(null)
  const [failure, setFailure] = useState(null)
  const [healthState, setHealthState] = useState(null)

  useEffect(() => {
    localStorage.setItem(MODE_KEY, mode)
  }, [mode])

  // One health fetch, shared: it drives both the status indicator and the
  // threshold track. Thresholds are read from the backend rather than baked in
  // here, because the ROC work is expected to move them.
  useEffect(() => {
    let cancelled = false
    health().then((outcome) => {
      if (!cancelled) setHealthState(outcome)
    })
    return () => {
      cancelled = true
    }
  }, [mode])

  const thresholds =
    healthState?.ok && healthState.data?.thresholds
      ? healthState.data.thresholds
      : DEFAULT_THRESHOLDS

  // Selecting a new file clears whatever was on screen. A verdict from the
  // previous file must never linger beside a new one.
  const handleFileSelected = useCallback(() => {
    setResult(null)
    setFailure(null)
  }, [])

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
      <header className="border-b border-rule">
        <div className="mx-auto flex max-w-[960px] flex-wrap items-center justify-between gap-x-6 gap-y-3 px-6 py-5">
          <div>
            <h1 className="text-[0.9375rem] leading-none font-semibold tracking-[0.16em] uppercase">
              VidTrust AI
            </h1>
            <p className="mt-1.5 text-[0.8125rem] text-ink-3">
              Machine-generated media detector
            </p>
          </div>
          <div className="flex items-center gap-5">
            <HealthIndicator mode={mode} state={healthState} />
            <ModeToggle mode={mode} onChange={setMode} disabled={busy} />
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[960px] px-6 pt-10 pb-24">
        <UploadZone
          onAnalyze={handleAnalyze}
          onFileSelected={handleFileSelected}
          busy={busy}
          progress={progress}
        />

        {failure && <ErrorState code={failure.code} message={failure.message} />}

        {result && !failure && (
          <ResultView result={result} thresholds={thresholds} />
        )}
      </main>
    </div>
  )
}

function ResultView({ result, thresholds }) {
  return (
    <div className="reveal mt-12">
      {/* Report header: what was measured, and what it cost to measure it. */}
      <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1 border-b border-rule pb-3">
        <p className="truncate text-sm font-medium" title={result.filename}>
          {result.filename}
        </p>
        <p className="label label-faint">
          {result.media_type} ·{' '}
          <span className="fig">{result.processing_time_ms}</span> ms · id{' '}
          <span className="fig">{result.id}</span>
        </p>
      </div>

      <div className="mt-10">
        <VerdictBlock
          verdict={result.verdict}
          confidence={result.confidence}
        />
        <ThresholdTrack
          confidence={result.confidence}
          thresholds={thresholds}
        />
      </div>

      <SignalLedger signals={result.signals} confidence={result.confidence} />

      <FrameTimeline
        frames={result.frames}
        modelAvailable={result.signals?.model?.available}
        thresholds={thresholds}
      />

      <RawResponse result={result} />
    </div>
  )
}
