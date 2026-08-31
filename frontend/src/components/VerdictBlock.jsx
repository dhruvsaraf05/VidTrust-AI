import { formatPercent } from '../lib/fusion'
import { styleForVerdict } from '../lib/verdict'

/**
 * The headline reading.
 *
 * Percentage, not the raw float — a viewer across a lecture hall reads 89%,
 * not 0.892. The precise figure lives in the ledger, where precision is the
 * point. The glyph is not decoration: it carries the verdict independently of
 * colour, so the reading survives a bad projector or a colour-blind viewer.
 */
export default function VerdictBlock({ verdict, confidence }) {
  const style = styleForVerdict(verdict)

  return (
    <section aria-labelledby="verdict-heading">
      <h2 id="verdict-heading" className="label">
        Verdict
      </h2>

      <div className="mt-4">
        <p
          className={`fig ${style.fg} font-medium leading-[0.85] tracking-[-0.03em]`}
          style={{ fontSize: 'clamp(3.25rem, 11vw, 5rem)' }}
        >
          {formatPercent(confidence)}
        </p>

        {/* The glyph rides with the word, not the number, where it reads as a
            status marker rather than a stray bullet. It is what carries the
            verdict when colour cannot — a bad projector, or a colour-blind
            viewer. */}
        <p
          className={`${style.fg} mt-3 flex items-center gap-2.5 text-[1.375rem] leading-tight font-semibold tracking-[0.12em] uppercase`}
        >
          <span aria-hidden="true" className="text-[0.9em] leading-none">
            {style.glyph}
          </span>
          {style.short}
        </p>
      </div>

      <p className="mt-4 max-w-prose text-ink-2">{style.reading}</p>
    </section>
  )
}
