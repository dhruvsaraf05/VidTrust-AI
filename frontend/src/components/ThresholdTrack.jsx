import { VERDICT_STYLE } from '../lib/verdict'

/**
 * The 0–1 confidence axis, with the two cut points drawn as the edges of three
 * zones rather than as bare numbers.
 *
 * Drawing UNCERTAIN as a BAND is the whole reason this exists. As a word in a
 * table it reads as a mysterious third label; as a region of the axis it reads
 * as what it is — the span where the system declines to answer.
 *
 * Zone boundaries come from GET /api/health. They are expected to move once
 * the thresholds are derived from an ROC curve, and this redraws itself when
 * they do.
 */
export default function ThresholdTrack({ confidence, thresholds }) {
  const low = thresholds.likely_real
  const high = thresholds.ai_generated

  const zones = [
    { key: 'LIKELY_REAL', from: 0, to: low },
    { key: 'UNCERTAIN', from: low, to: high },
    { key: 'AI_GENERATED', from: high, to: 1 },
  ]

  const marker = Math.min(Math.max(confidence, 0), 1)

  return (
    <section className="mt-8" aria-labelledby="track-heading">
      <h3 id="track-heading" className="label">
        Confidence scale
      </h3>

      <div className="relative mt-3">
        <div className="flex h-12 overflow-hidden rounded-[3px] border border-rule">
          {zones.map((zone) => {
            const style = VERDICT_STYLE[zone.key]
            return (
              <div
                key={zone.key}
                className="flex items-center justify-center overflow-hidden border-r border-rule last:border-r-0"
                style={{
                  width: `${(zone.to - zone.from) * 100}%`,
                  backgroundColor: style.tint,
                }}
              >
                <span
                  className="truncate px-2 text-[0.8125rem] leading-none font-medium tracking-[0.08em] uppercase"
                  style={{ color: style.fill }}
                >
                  {style.short}
                </span>
              </div>
            )
          })}
        </div>

        {/* Needle. Sits at the fused confidence; the figure itself is the
            headline above, so it is not repeated here. */}
        <div
          className="pointer-events-none absolute top-0 bottom-0 transition-[left] duration-500"
          style={{ left: `${marker * 100}%` }}
        >
          <div className="absolute top-0 bottom-0 -left-px w-0.5 bg-ink" />
          <div
            className="absolute -top-1.5 -left-[5px] h-0 w-0"
            style={{
              borderLeft: '5px solid transparent',
              borderRight: '5px solid transparent',
              borderTop: '6px solid var(--color-ink)',
            }}
          />
        </div>
      </div>

      <div className="relative mt-2 h-4">
        {[
          { at: 0, text: '0.00' },
          { at: low, text: low.toFixed(2) },
          { at: high, text: high.toFixed(2) },
          { at: 1, text: '1.00' },
        ].map((tick) => (
          <span
            key={tick.at}
            className="fig absolute text-[0.8125rem] text-ink-3"
            style={{
              left: `${tick.at * 100}%`,
              transform:
                tick.at === 0
                  ? 'none'
                  : tick.at === 1
                    ? 'translateX(-100%)'
                    : 'translateX(-50%)',
            }}
          >
            {tick.text}
          </span>
        ))}
      </div>
    </section>
  )
}
