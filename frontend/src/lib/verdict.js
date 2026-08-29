/**
 * Verdict thresholds and their visual treatment.
 *
 * These mirror THRESHOLD_AI_GENERATED / THRESHOLD_LIKELY_REAL in
 * backend/config.py. The backend is authoritative -- it sends `verdict` in the
 * response and the UI should render that field, not recompute it. `verdictFor`
 * exists for mock data and for styling previews, NOT to second-guess the API.
 *
 * If the PRD's threshold-selection work (D7) moves these, update both files.
 * GET /api/health reports the live values under `thresholds`.
 */

export const THRESHOLDS = {
  AI_GENERATED: 0.65,
  LIKELY_REAL: 0.35,
}

export const VERDICTS = ['AI_GENERATED', 'UNCERTAIN', 'LIKELY_REAL']

/** Only for mock/preview. Prefer the `verdict` field from the API. */
export function verdictFor(confidence) {
  if (confidence >= THRESHOLDS.AI_GENERATED) return 'AI_GENERATED'
  if (confidence <= THRESHOLDS.LIKELY_REAL) return 'LIKELY_REAL'
  return 'UNCERTAIN'
}

/**
 * UNCERTAIN is styled now, deliberately, rather than discovered during the
 * demo. It is not a degraded AI_GENERATED -- it is the system declining to
 * answer, which the PRD treats as a legitimate third outcome. It reads as
 * neutral amber, not as a failure state.
 */
export const VERDICT_STYLE = {
  AI_GENERATED: {
    label: 'AI-generated',
    text: 'text-ai',
    bg: 'bg-ai-soft',
    border: 'border-ai/30',
    dot: 'bg-ai',
  },
  LIKELY_REAL: {
    label: 'Likely real',
    text: 'text-real',
    bg: 'bg-real-soft',
    border: 'border-real/30',
    dot: 'bg-real',
  },
  UNCERTAIN: {
    label: 'Uncertain',
    text: 'text-uncertain',
    bg: 'bg-uncertain-soft',
    border: 'border-uncertain/30',
    dot: 'bg-uncertain',
  },
}

export function styleForVerdict(verdict) {
  return VERDICT_STYLE[verdict] ?? VERDICT_STYLE.UNCERTAIN
}
