/**
 * Verdict thresholds and their visual treatment.
 *
 * Thresholds are READ FROM GET /api/health, not hardcoded here. The PRD's
 * threshold-selection work derives them from an ROC curve, so they are
 * expected to move; anything that bakes 0.65/0.35 into the UI would silently
 * disagree with the backend the moment they do.
 *
 * DEFAULT_THRESHOLDS is a last-resort fallback for when health cannot be
 * reached at all (mock mode with the server stopped). It mirrors
 * backend/config.py at time of writing and is the one place these numbers
 * appear on the client.
 */

export const DEFAULT_THRESHOLDS = {
  ai_generated: 0.65,
  likely_real: 0.35,
}

export const VERDICTS = ['LIKELY_REAL', 'UNCERTAIN', 'AI_GENERATED']

/** Only for previews. The API sends `verdict`; render that. */
export function verdictFor(confidence, thresholds = DEFAULT_THRESHOLDS) {
  if (confidence >= thresholds.ai_generated) return 'AI_GENERATED'
  if (confidence <= thresholds.likely_real) return 'LIKELY_REAL'
  return 'UNCERTAIN'
}

/**
 * UNCERTAIN is not a degraded AI_GENERATED. It is the system declining to
 * answer, which the project treats as a legitimate third outcome — so it gets
 * its own hue and its own wording, and the threshold track draws it as a band
 * rather than a boundary.
 *
 * Every verdict carries a glyph as well as a colour. Colour never carries
 * meaning on its own anywhere in this interface.
 */
export const VERDICT_STYLE = {
  AI_GENERATED: {
    label: 'AI-generated',
    short: 'AI-GENERATED',
    glyph: '▲',
    reading: 'Signals agree this was machine-generated.',
    fg: 'text-crimson',
    bgTint: 'bg-crimson-tint',
    border: 'border-crimson',
    fill: 'var(--color-crimson)',
    tint: 'var(--color-crimson-tint)',
  },
  UNCERTAIN: {
    label: 'Uncertain',
    short: 'UNCERTAIN',
    glyph: '◆',
    reading: 'Signals disagree. The system is declining to answer.',
    fg: 'text-ochre',
    bgTint: 'bg-ochre-tint',
    border: 'border-ochre',
    fill: 'var(--color-ochre)',
    tint: 'var(--color-ochre-tint)',
  },
  LIKELY_REAL: {
    label: 'Likely real',
    short: 'LIKELY REAL',
    glyph: '●',
    reading: 'Signals agree this was captured, not generated.',
    fg: 'text-teal',
    bgTint: 'bg-teal-tint',
    border: 'border-teal',
    fill: 'var(--color-teal)',
    tint: 'var(--color-teal-tint)',
  },
}

export function styleForVerdict(verdict) {
  return VERDICT_STYLE[verdict] ?? VERDICT_STYLE.UNCERTAIN
}
