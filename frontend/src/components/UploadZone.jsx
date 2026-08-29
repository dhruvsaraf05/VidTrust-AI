import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { FileVideo, Image as ImageIcon, Upload, X } from 'lucide-react'

import {
  ACCEPT_ATTR,
  formatBytes,
  mediaTypeOf,
  validateFile,
} from '../lib/validation'

/**
 * Stage hints shown while the server works.
 *
 * These are INDICATIVE, not reported by the backend -- the API returns one
 * response at the end and tells us nothing about its internal progress. They
 * are ordered to match what the pipeline actually does so they aren't
 * misleading, and the elapsed counter beside them is the real measurement.
 * Do not present these as live server state.
 */
const VIDEO_HINTS = [
  'Sampling frames…',
  'Running classifier per frame…',
  'Reading container metadata…',
  'Analysing frequency…',
  'Aggregating signals…',
]
const IMAGE_HINTS = ['Running classifier…', 'Reading metadata…', 'Analysing frequency…']

/**
 * Progress line for an in-flight request.
 *
 * Mounted only while busy, so its clock starts at zero on every run without
 * anything having to reset it. The elapsed seconds are the honest measurement;
 * the hint beside them is indicative only (see the note on the hint arrays).
 */
function ProgressLine({ hints, uploading, percent }) {
  const [elapsed, setElapsed] = useState(0)

  useEffect(() => {
    const startedAt = Date.now()
    const timer = setInterval(() => {
      setElapsed((Date.now() - startedAt) / 1000)
    }, 100)
    return () => clearInterval(timer)
  }, [])

  const hint = hints[Math.min(Math.floor(elapsed / 3), hints.length - 1)]

  return (
    <div className="mt-3">
      <div className="h-px w-full bg-rule">
        <div
          className="h-px bg-accent transition-[width] duration-150"
          style={{ width: uploading ? `${percent}%` : '100%' }}
        />
      </div>
      <p className="micro mt-2 flex items-center justify-between gap-3">
        <span>{uploading ? `Uploading ${percent}%` : hint}</span>
        <span className="num text-ink-faint">{elapsed.toFixed(1)}s</span>
      </p>
    </div>
  )
}

export default function UploadZone({ onAnalyze, busy, progress }) {
  const [file, setFile] = useState(null)
  const [error, setError] = useState(null)
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef(null)

  const kind = file ? mediaTypeOf(file.name) : null

  // Derived, not stored: avoids a setState-in-effect reset when the file is
  // cleared. Object URLs must still be revoked or the tab leaks memory.
  const previewUrl = useMemo(
    () => (file ? URL.createObjectURL(file) : null),
    [file],
  )
  useEffect(() => {
    if (!previewUrl) return
    return () => URL.revokeObjectURL(previewUrl)
  }, [previewUrl])

  const accept = useCallback((candidate) => {
    if (!candidate) return
    const result = validateFile(candidate)
    if (!result.ok) {
      setError(result.message)
      setFile(null)
      return
    }
    setError(null)
    setFile(candidate)
  }, [])

  const onDrop = (event) => {
    event.preventDefault()
    setDragging(false)
    if (busy) return
    accept(event.dataTransfer.files?.[0])
  }

  const clear = () => {
    setFile(null)
    setError(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  const hints = kind === 'video' ? VIDEO_HINTS : IMAGE_HINTS
  const uploading = busy && progress?.phase === 'uploading' && progress.percent < 100

  return (
    <section className="w-full">
      <div
        onDragOver={(event) => {
          event.preventDefault()
          if (!busy) setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={[
          'relative rounded-sm border border-dashed bg-surface transition-colors',
          dragging ? 'border-accent bg-accent-soft' : 'border-rule-strong',
          busy ? 'opacity-60' : '',
        ].join(' ')}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT_ATTR}
          className="hidden"
          onChange={(event) => accept(event.target.files?.[0])}
        />

        {!file ? (
          <button
            type="button"
            disabled={busy}
            onClick={() => inputRef.current?.click()}
            className="flex w-full cursor-pointer flex-col items-center gap-3 px-6 py-14 text-center disabled:cursor-not-allowed"
          >
            <Upload className="h-6 w-6 text-ink-faint" strokeWidth={1.5} />
            <span className="text-sm text-ink">
              Drop a file here, or{' '}
              <span className="text-accent underline underline-offset-4">browse</span>
            </span>
            <span className="micro">
              JPG · PNG · WEBP · MP4 · MOV · AVI — max 50 MB
            </span>
          </button>
        ) : (
          <div className="flex items-start gap-4 p-4">
            <div className="flex h-20 w-20 shrink-0 items-center justify-center overflow-hidden rounded-sm border border-rule bg-surface-sunken">
              {kind === 'image' && previewUrl ? (
                <img
                  src={previewUrl}
                  alt=""
                  className="h-full w-full object-cover"
                />
              ) : kind === 'video' && previewUrl ? (
                <video
                  src={previewUrl}
                  muted
                  playsInline
                  preload="metadata"
                  className="h-full w-full object-cover"
                />
              ) : kind === 'video' ? (
                <FileVideo className="h-6 w-6 text-ink-faint" strokeWidth={1.5} />
              ) : (
                <ImageIcon className="h-6 w-6 text-ink-faint" strokeWidth={1.5} />
              )}
            </div>

            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-medium text-ink" title={file.name}>
                {file.name}
              </p>
              <p className="micro mt-1.5">
                {kind} · <span className="num">{formatBytes(file.size)}</span>
              </p>

              {busy && (
                <ProgressLine
                  hints={hints}
                  uploading={uploading}
                  percent={progress?.percent ?? 0}
                />
              )}
            </div>

            {!busy && (
              <button
                type="button"
                onClick={clear}
                aria-label="Remove file"
                className="shrink-0 cursor-pointer rounded-sm p-1 text-ink-faint hover:bg-surface-sunken hover:text-ink"
              >
                <X className="h-4 w-4" strokeWidth={1.5} />
              </button>
            )}
          </div>
        )}
      </div>

      {error && (
        <p className="mt-2 border-l-2 border-ai bg-ai-soft px-3 py-2 text-sm text-ai">
          {error}
        </p>
      )}

      <div className="mt-4 flex items-center gap-3">
        <button
          type="button"
          disabled={!file || busy}
          onClick={() => file && onAnalyze(file)}
          className="cursor-pointer rounded-sm bg-accent px-5 py-2 text-sm font-medium text-white transition-colors hover:bg-accent-hover disabled:cursor-not-allowed disabled:bg-rule-strong"
        >
          {busy ? 'Analysing…' : 'Analyze'}
        </button>
        {file && !busy && (
          <span className="micro">
            Server enforces its own limits — client checks are a convenience
          </span>
        )}
      </div>
    </section>
  )
}
