import { useCallback, useEffect, useState } from 'react'

import { MODES, analyze, analyzeUrl, health } from './api/client'
import ErrorState from './components/ErrorState'
import FrameTimeline from './components/FrameTimeline'
import HealthIndicator from './components/HealthIndicator'
import ModeToggle from './components/ModeToggle'
import RawResponse from './components/RawResponse'
import ThemeSwitcher from './components/ThemeSwitcher' // TEMPORARY
import SignalLedger from './components/SignalLedger'
import ThresholdTrack from './components/ThresholdTrack'
import UploadZone from './components/UploadZone'
import UrlZone from './components/UrlZone'
import VerdictBlock from './components/VerdictBlock'
import { DEFAULT_THRESHOLDS } from './lib/verdict'

const MODE_KEY = 'vidtrust.mode'
const SOURCE_KEY = 'vidtrust.source'
const ABOUT_KEY = 'vidtrust.about-open'

export default function App() {
  // null means "no explicit choice yet, and not detected yet" -- it is
  // resolved below, once, by probing the backend. Mock must never be the
  // silent default on a fresh visit: it always returns one canned fixture
  // (confidence 0.2504) regardless of what file is given to it, and a first-
  // time viewer with the backend running has no way to know a real image
  // was never actually analysed. So an UNSET preference resolves to Live if
  // the backend answers, Mock only if it doesn't. An EXPLICIT choice -- the
  // user toggled it themselves -- is always respected and never overridden.
  const [mode, setMode] = useState(() => localStorage.getItem(MODE_KEY))
  const [detectingMode, setDetectingMode] = useState(mode === null)
  const resolvedMode = mode ?? MODES.LIVE

  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState(null)
  const [result, setResult] = useState(null)
  const [resultSource, setResultSource] = useState(null) // mode the last result came from
  const [failure, setFailure] = useState(null)
  const [healthState, setHealthState] = useState(null)
  // 'file' or 'url'. Both feed the identical result components.
  const [inputMode, setInputMode] = useState(
    () => localStorage.getItem(SOURCE_KEY) ?? 'file',
  )

  useEffect(() => {
    localStorage.setItem(SOURCE_KEY, inputMode)
  }, [inputMode])

  useEffect(() => {
    // Only persist a REAL choice. Writing the optimistic pre-detection value
    // would turn "not decided yet" into a stored preference by accident.
    if (mode !== null) localStorage.setItem(MODE_KEY, mode)
  }, [mode])

  // Runs once. If there is no stored preference, ping the backend and
  // resolve Live/Mock from whether it answers -- see the note above.
  useEffect(() => {
    if (mode !== null) return
    let cancelled = false
    health().then((outcome) => {
      if (cancelled) return
      setHealthState(outcome)
      setMode(outcome.ok ? MODES.LIVE : MODES.MOCK)
      setDetectingMode(false)
    })
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // One health fetch, shared: it drives both the status indicator and the
  // threshold track. Thresholds are read from the backend rather than baked in
  // here, because the ROC work is expected to move them.
  useEffect(() => {
    if (mode === null) return // covered by the detection effect above
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

  async function runAnalysis(work, source) {
    setBusy(true)
    setProgress(null)
    setResult(null)
    setResultSource(null)
    setFailure(null)

    const outcome = await work()

    if (outcome.ok) {
      setResult(outcome.data)
      setResultSource(source) // stamped so a fixture result can't pass as real
    } else {
      // Branch on the body's `error` field, never on the HTTP status.
      setFailure({ code: outcome.error, message: outcome.message })
    }
    setBusy(false)
  }

  const handleAnalyze = (file) =>
    runAnalysis(
      () => analyze(file, { mode: resolvedMode, onProgress: setProgress }),
      resolvedMode,
    )

  // Link mode always calls the live backend -- there is no URL fixture.
  const handleAnalyzeUrl = (url) => runAnalysis(() => analyzeUrl(url), MODES.LIVE)

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
            <HealthIndicator mode={resolvedMode} state={healthState} />
            <ModeToggle mode={resolvedMode} onChange={setMode} disabled={busy} />
            <ThemeSwitcher />
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[960px] px-6 pt-10 pb-24">
        <div className="mb-5 flex items-center gap-0.5 rounded-[3px] border border-rule bg-panel p-0.5 w-fit">
          {[
            { value: 'file', label: 'Upload file' },
            { value: 'url', label: 'Paste link' },
          ].map((tab) => (
            <button
              key={tab.value}
              type="button"
              disabled={busy}
              aria-pressed={inputMode === tab.value}
              onClick={() => {
                setInputMode(tab.value)
                handleFileSelected()
              }}
              className={[
                'label cursor-pointer rounded-[2px] px-4 py-2 transition-colors disabled:cursor-not-allowed',
                inputMode === tab.value
                  ? 'bg-ink text-panel'
                  : 'hover:text-ink disabled:hover:text-ink-2',
              ].join(' ')}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {inputMode === 'file' ? (
          <UploadZone
            onAnalyze={handleAnalyze}
            onFileSelected={handleFileSelected}
            busy={busy || detectingMode}
            progress={progress}
          />
        ) : (
          <UrlZone
            onAnalyze={handleAnalyzeUrl}
            onInputChanged={handleFileSelected}
            busy={busy || detectingMode}
            live
          />
        )}

        {failure && <ErrorState code={failure.code} message={failure.message} />}

        {result && !failure && (
          <ResultView
            result={result}
            thresholds={thresholds}
            isFixture={resultSource === MODES.MOCK}
            onSwitchToLive={() => setMode(MODES.LIVE)}
          />
        )}

        {/* Below the tool itself, always: the argument for what it does,
            not blocking the action a visitor came here to take. */}
        <div className={result || failure ? 'mt-16' : 'mt-10'}>
          <AboutPanel />
        </div>
      </main>
    </div>
  )
}

/**
 * Project context, collapsed after the first read.
 *
 * A page that is only an upload zone and a result reads as a bare tool with
 * no argument behind it. This is the argument: what is being measured, the
 * three signals and their weights, and the renormalisation rule that is the
 * actual thesis of the design (see SignalLedger and REPORT.md section 2).
 *
 * Deliberately excludes performance numbers. Every evaluated figure in
 * REPORT.md carries a caveat about dataset confounds (zero EXIF, an FFHQ
 * content confound, a resolution asymmetry); quoting one here, stripped of
 * that context, would overclaim exactly what the evaluation work exists to
 * avoid. This panel documents the ARCHITECTURE, not a number.
 */
function AboutPanel() {
  const [open, setOpen] = useState(
    () => (localStorage.getItem(ABOUT_KEY) ?? 'open') === 'open',
  )

  useEffect(() => {
    localStorage.setItem(ABOUT_KEY, open ? 'open' : 'closed')
  }, [open])

  return (
    <section className="panel mb-8">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full cursor-pointer items-center justify-between gap-4 px-5 py-3.5 text-left"
      >
        <span className="label">About this tool</span>
        <span className="label label-faint" aria-hidden="true">
          {open ? 'Hide ▴' : 'Show ▾'}
        </span>
      </button>

      {open && (
        <div className="border-t border-rule px-5 py-5">
          <p className="max-w-prose text-sm text-ink">
            VidTrust AI answers one question about an image or video: was it{' '}
            <strong className="font-semibold">captured by a camera</strong>, or{' '}
            <strong className="font-semibold">produced by a generative model</strong>{' '}
            such as Midjourney, SDXL, DALL·E, Sora or Veo? It does not detect
            deepfake face-swaps or judge content quality — it measures
            provenance.
          </p>

          <p className="label mt-6 mb-2">Three independent signals, fused</p>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[520px] border-collapse text-sm">
              <thead>
                <tr className="border-b border-rule text-left">
                  <th className="label label-faint py-1.5 pr-4 font-normal">Signal</th>
                  <th className="label label-faint py-1.5 pr-4 font-normal">Weight</th>
                  <th className="label label-faint py-1.5 font-normal">Evidence</th>
                </tr>
              </thead>
              <tbody className="text-ink-2">
                <tr className="border-b border-rule">
                  <td className="py-2 pr-4 text-ink">Classifier</td>
                  <td className="fig py-2 pr-4">0.60</td>
                  <td className="py-2">
                    <code className="fig text-[0.8125rem]">Organika/sdxl-detector</code>,
                    used as published — no training or fine-tuning
                  </td>
                </tr>
                <tr className="border-b border-rule">
                  <td className="py-2 pr-4 text-ink">Provenance</td>
                  <td className="fig py-2 pr-4">0.25</td>
                  <td className="py-2">EXIF / XMP / C2PA generator fingerprints</td>
                </tr>
                <tr>
                  <td className="py-2 pr-4 text-ink">Frequency</td>
                  <td className="fig py-2 pr-4">0.15</td>
                  <td className="py-2">FFT high-frequency energy ratio (provisional)</td>
                </tr>
              </tbody>
            </table>
          </div>

          <p className="label mt-6 mb-2">The rule that matters most</p>
          <p className="max-w-prose text-sm text-ink-2">
            A signal that cannot run — no EXIF on a re-encoded upload, a dead
            classifier — is <strong className="font-semibold text-ink">removed</strong>{' '}
            from the average, and the remaining weights are rescaled to fill the
            gap. It is never scored as 0.0. A missing signal must never be able
            to drag a verdict toward &ldquo;real&rdquo; — see the{' '}
            <span className="fig">▸</span> notation in the weight column below
            once a result is in.
          </p>

          <p className="label mt-6 mb-2">Honest limits</p>
          <ul className="max-w-prose list-inside list-disc space-y-1 text-sm text-ink-2">
            <li>Weights and verdict thresholds above are hand-chosen, not fitted.</li>
            <li>
              The frequency signal is a provisional heuristic — its score means
              &ldquo;how high-frequency is this image&rdquo;, not yet a
              measured AI/real boundary.
            </li>
            <li>
              Social platforms and third-party URLs re-encode media on upload,
              stripping provenance — an unavailable metadata signal there is
              expected, not a failure.
            </li>
          </ul>

          <p className="label label-faint mt-6">
            Full methodology, evaluation and ablation results are in
            REPORT.md and README.md in the project repository.
          </p>
        </div>
      )}
    </section>
  )
}

function ResultView({ result, thresholds, isFixture, onSwitchToLive }) {
  return (
    <div className="reveal mt-12">
      {/* This is what a genuine misclassification looked like from the
          outside once: a real image, a canned 25% "likely real". The fixture
          exists so the UI can be built with no server, but nothing below it
          may be mistaken for a real verdict -- hence this banner rather than
          only the small header dot, which is easy to miss entirely. */}
      {isFixture && (
        <div className="mb-6 flex flex-wrap items-center justify-between gap-3 border-l-2 border-ochre bg-ochre-tint px-4 py-3">
          <p className="text-sm text-ink">
            <strong className="font-semibold">Fixture response.</strong> This
            is a captured example, not an analysis of the file you gave it —
            the classifier never ran on it.
          </p>
          {onSwitchToLive && (
            <button
              type="button"
              onClick={onSwitchToLive}
              className="label shrink-0 cursor-pointer rounded-[2px] border border-ochre px-3 py-1.5 text-ochre transition-colors hover:bg-ochre hover:text-panel"
            >
              Switch to Live
            </button>
          )}
        </div>
      )}

      {/* Report header: what was measured, and what it cost to measure it. */}
      <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1 border-b border-rule pb-3">
        <p className="truncate text-sm font-medium" title={result.filename}>
          {result.filename}
        </p>
        <p className="label label-faint">
          {result.source ? (
            <>
              {result.source.platform}
              {result.source.duration_s != null && (
                <> · <span className="fig">{result.source.duration_s}</span>s</>
              )}{' '}
              ·{' '}
            </>
          ) : null}
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
      />

      <RawResponse result={result} />
    </div>
  )
}
