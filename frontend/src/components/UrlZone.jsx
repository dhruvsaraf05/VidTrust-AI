import { useEffect, useState } from 'react'
import { Link2 } from 'lucide-react'

/**
 * Link input. The counterpart to UploadZone, feeding the same result
 * components -- /api/analyze-url returns the /api/analyze shape plus `source`.
 */
function ProgressLine({ hints }) {
  const [elapsed, setElapsed] = useState(0)

  useEffect(() => {
    const startedAt = Date.now()
    const timer = setInterval(() => {
      setElapsed((Date.now() - startedAt) / 1000)
    }, 100)
    return () => clearInterval(timer)
  }, [])

  const hint = hints[Math.min(Math.floor(elapsed / 4), hints.length - 1)]

  return (
    <div className="mt-4">
      <div className="h-px w-full bg-rule">
        <div className="h-px w-full animate-pulse bg-accent" />
      </div>
      <p className="label mt-2 flex items-center justify-between gap-3">
        <span>{hint}</span>
        <span className="fig label-faint">{elapsed.toFixed(1)}s</span>
      </p>
    </div>
  )
}

// Indicative only -- the API reports no intermediate progress. Ordered to
// match what actually happens, with the elapsed clock as the real measurement.
const HINTS = [
  'Reading the page',
  'Fetching media',
  'Sampling frames',
  'Scoring each frame',
  'Fusing signals',
]

export default function UrlZone({ onAnalyze, onInputChanged, busy, live }) {
  const [url, setUrl] = useState('')

  return (
    <section aria-label="Analyse a link">
      <div className="rounded-[3px] border border-rule bg-panel p-4">
        <label htmlFor="media-url" className="label flex items-center gap-2">
          <Link2 className="h-3.5 w-3.5" strokeWidth={1.5} />
          Public video link
        </label>

        <input
          id="media-url"
          type="url"
          inputMode="url"
          placeholder="https://www.youtube.com/watch?v=..."
          value={url}
          disabled={busy}
          onChange={(event) => {
            setUrl(event.target.value)
            onInputChanged?.()
          }}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && url.trim() && !busy) onAnalyze(url.trim())
          }}
          className="mt-3 w-full rounded-[2px] border border-rule bg-panel-sunk px-3 py-2.5 text-sm text-ink placeholder:text-ink-3 disabled:opacity-60"
        />

        <p className="label label-faint mt-3">
          Public links only · max 60 s · capped at 720p · no login or cookies
        </p>

        {busy && <ProgressLine hints={HINTS} />}
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-3">
        <button
          type="button"
          disabled={!url.trim() || busy}
          onClick={() => onAnalyze(url.trim())}
          className="cursor-pointer rounded-[3px] bg-accent px-6 py-2.5 text-sm font-medium tracking-wide text-panel transition-colors hover:bg-accent-hi disabled:cursor-not-allowed disabled:bg-rule-2"
        >
          {busy ? 'Analysing' : 'Analyze'}
        </button>
        {!live && !busy && (
          <span className="label label-faint">
            Link mode always calls the live backend
          </span>
        )}
      </div>
    </section>
  )
}
