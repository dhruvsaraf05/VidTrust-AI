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
 * INDICATIVE ONLY — the API returns a single response at the end and reports
 * nothing about its internal progress. They are ordered to match what the
 * pipeline actually does so they are not misleading, and the elapsed clock
 * beside them is the real measurement. Never present these as live state.
 */
const VIDEO_HINTS = [
  'Sampling frames',
  'Scoring each frame',
  'Reading container metadata',
  'Measuring frequency',
  'Fusing signals',
]
const IMAGE_HINTS = ['Scoring image', 'Reading metadata', 'Measuring frequency']

/**
 * Mounted only while a request is in flight, so its clock starts at zero on
 * every run with nothing having to reset it.
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
      <p className="label mt-2 flex items-center justify-between gap-3">
        <span>{uploading ? `Uploading ${percent}%` : hint}</span>
        <span className="fig label-faint">{elapsed.toFixed(1)}s</span>
      </p>
    </div>
  )
}

export default function UploadZone({
  onAnalyze,
  onFileSelected,
  busy,
  progress,
}) {
  const [file, setFile] = useState(null)
  const [error, setError] = useState(null)
  const [dragging, setDragging] = useState(false)
  // A corrupt file is a demo scenario, and the browser's broken-image glyph is
  // not the way to present one. Fall back to the plain type icon.
  const [previewBroken, setPreviewBroken] = useState(false)
  const inputRef = useRef(null)

  const kind = file ? mediaTypeOf(file.name) : null

  const previewUrl = useMemo(
    () => (file ? URL.createObjectURL(file) : null),
    [file],
  )
  useEffect(() => {
    if (!previewUrl) return
    return () => URL.revokeObjectURL(previewUrl)
  }, [previewUrl])

  const accept = useCallback(
    (candidate) => {
      if (!candidate) return
      // Any new selection clears the previous verdict, valid or not.
      onFileSelected?.()

      const outcome = validateFile(candidate)
      if (!outcome.ok) {
        setError(outcome.message)
        setFile(null)
        return
      }
      setError(null)
      setPreviewBroken(false)
      setFile(candidate)
    },
    [onFileSelected],
  )

  const clear = () => {
    setFile(null)
    setError(null)
    setPreviewBroken(false)
    onFileSelected?.()
    if (inputRef.current) inputRef.current.value = ''
  }

  const hints = kind === 'video' ? VIDEO_HINTS : IMAGE_HINTS
  const uploading =
    busy && progress?.phase === 'uploading' && progress.percent < 100

  return (
    <section aria-label="Choose a file">
      <div
        onDragOver={(event) => {
          event.preventDefault()
          if (!busy) setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault()
          setDragging(false)
          if (!busy) accept(event.dataTransfer.files?.[0])
        }}
        className={[
          'rounded-[3px] border bg-panel transition-colors',
          dragging ? 'border-accent bg-accent-tint' : 'border-rule',
          file ? '' : 'border-dashed',
          busy ? 'opacity-70' : '',
        ].join(' ')}
      >
        {/* tabIndex -1: the visible button below is the control. Leaving the
            hidden input in the tab order gives two stops for one action, and
            the first of them shows no focus ring at all. */}
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT_ATTR}
          tabIndex={-1}
          className="sr-only"
          onChange={(event) => accept(event.target.files?.[0])}
        />

        {!file ? (
          <button
            type="button"
            disabled={busy}
            onClick={() => inputRef.current?.click()}
            className="flex w-full cursor-pointer flex-col items-center gap-3 px-6 py-12 text-center disabled:cursor-not-allowed"
          >
            <Upload className="h-5 w-5 text-ink-3" strokeWidth={1.5} />
            <span className="text-sm">
              Drop a file here, or{' '}
              <span className="text-accent underline underline-offset-4">
                browse
              </span>
            </span>
            <span className="label label-faint">
              JPG · PNG · WEBP · MP4 · MOV · AVI — max 50 MB
            </span>
          </button>
        ) : (
          <div className="flex items-start gap-4 p-4">
            <div className="flex h-16 w-16 shrink-0 items-center justify-center overflow-hidden rounded-[2px] border border-rule bg-panel-sunk">
              {previewBroken ? (
                kind === 'video' ? (
                  <FileVideo className="h-5 w-5 text-ink-3" strokeWidth={1.5} />
                ) : (
                  <ImageIcon className="h-5 w-5 text-ink-3" strokeWidth={1.5} />
                )
              ) : kind === 'image' ? (
                <img
                  src={previewUrl}
                  alt=""
                  onError={() => setPreviewBroken(true)}
                  className="h-full w-full object-cover"
                />
              ) : kind === 'video' ? (
                <video
                  src={previewUrl}
                  muted
                  playsInline
                  preload="metadata"
                  onError={() => setPreviewBroken(true)}
                  className="h-full w-full object-cover"
                />
              ) : (
                <ImageIcon className="h-5 w-5 text-ink-3" strokeWidth={1.5} />
              )}
            </div>

            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-medium" title={file.name}>
                {file.name}
              </p>
              <p className="label label-faint mt-1.5 flex items-center gap-1.5">
                {kind === 'video' ? (
                  <FileVideo className="h-3.5 w-3.5" strokeWidth={1.5} />
                ) : (
                  <ImageIcon className="h-3.5 w-3.5" strokeWidth={1.5} />
                )}
                {kind} · <span className="fig">{formatBytes(file.size)}</span>
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
                className="shrink-0 cursor-pointer rounded-[2px] p-1 text-ink-3 hover:bg-panel-sunk hover:text-ink"
              >
                <X className="h-4 w-4" strokeWidth={1.5} />
              </button>
            )}
          </div>
        )}
      </div>

      {error && (
        <p className="mt-2 border-l-2 border-crimson bg-crimson-tint px-4 py-2.5 text-sm text-ink">
          {error}
        </p>
      )}

      <div className="mt-4">
        <button
          type="button"
          disabled={!file || busy}
          onClick={() => file && onAnalyze(file)}
          className="cursor-pointer rounded-[3px] bg-accent px-6 py-2.5 text-sm font-medium tracking-wide text-panel transition-colors hover:bg-accent-hi disabled:cursor-not-allowed disabled:bg-rule-2"
        >
          {busy ? 'Analysing' : 'Analyze'}
        </button>
      </div>
    </section>
  )
}
