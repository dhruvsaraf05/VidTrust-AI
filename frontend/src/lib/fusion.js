/**
 * The fusion arithmetic, made explicit so the UI can show its working.
 *
 * The backend fuses three signals with a weighted average, and when a signal
 * cannot run it is REMOVED from the average and the remaining weights are
 * renormalised — it is never scored as 0.0. That distinction is the single
 * most important behaviour in the system: a missing signal must not be able to
 * drag a verdict toward "real".
 *
 * That invariant is invisible in a table of numbers, so this module recomputes
 * it client-side purely for display:
 *
 *     effective weight = nominal weight / (sum of available nominal weights)
 *     contribution     = score x effective weight
 *     fused confidence = sum of contributions
 *
 * The backend remains the source of truth — `confidence` from the API is what
 * gets displayed as the total. These numbers reconcile with it to 4dp; they
 * exist to show HOW that number was reached, not to second-guess it.
 */

export const SIGNAL_ORDER = ['model', 'metadata', 'frequency']

/**
 * @returns rows in a fixed display order, each carrying its nominal weight,
 * its effective (renormalised) weight, and its contribution to the total.
 */
export function buildLedger(signals) {
  const rows = SIGNAL_ORDER.filter((key) => signals?.[key]).map((key) => ({
    key,
    ...signals[key],
  }))

  const availableWeight = rows
    .filter((row) => row.available)
    .reduce((sum, row) => sum + row.weight, 0)

  return rows.map((row) => {
    const effectiveWeight =
      row.available && availableWeight > 0 ? row.weight / availableWeight : null

    return {
      ...row,
      nominalWeight: row.weight,
      effectiveWeight,
      contribution: effectiveWeight === null ? 0 : row.score * effectiveWeight,
      // True when this signal had to absorb weight from an absent one.
      widened:
        effectiveWeight !== null &&
        Math.abs(effectiveWeight - row.weight) > 0.0005,
    }
  })
}

export function unavailableRows(rows) {
  return rows.filter((row) => !row.available)
}

/** Sum of contributions. Reconciles with the API's `confidence`. */
export function ledgerSum(rows) {
  return rows.reduce((sum, row) => sum + row.contribution, 0)
}

/**
 * Contributions rounded so the column visibly adds up to the displayed total.
 *
 * Rounding each contribution independently can leave a column that does not
 * sum to its own total — 0.000 + 0.003 against a stated 0.004. On a page whose
 * entire argument is that the verdict is a transparent sum, a column that does
 * not add up is the one flaw a viewer will find, and it undermines everything
 * around it.
 *
 * This applies the largest-remainder method: floor every contribution at the
 * display precision, then hand the leftover thousandths to the rows with the
 * biggest discarded fractions. Each displayed figure stays within one unit of
 * the last decimal place of its true value, and the column reconciles exactly.
 * The unrounded numbers are always in the raw response.
 *
 * @returns Map of signal key -> displayed contribution
 */
export function displayContributions(rows, total, decimals = 3) {
  const scale = 10 ** decimals
  const available = rows.filter((row) => row.available)

  const parts = available.map((row) => {
    const exact = row.contribution * scale
    const floor = Math.floor(exact)
    return { key: row.key, units: floor, remainder: exact - floor }
  })

  let deficit =
    Math.round(total * scale) - parts.reduce((sum, part) => sum + part.units, 0)

  const byRemainder = [...parts].sort((a, b) => b.remainder - a.remainder)
  let cursor = 0
  while (deficit > 0 && byRemainder.length > 0) {
    byRemainder[cursor % byRemainder.length].units += 1
    cursor += 1
    deficit -= 1
  }
  while (deficit < 0 && byRemainder.length > 0) {
    const candidate = byRemainder[cursor % byRemainder.length]
    if (candidate.units > 0) {
      candidate.units -= 1
      deficit += 1
    }
    cursor += 1
  }

  return new Map(parts.map((part) => [part.key, part.units / scale]))
}

/**
 * Weights render in the ledger's own notation: a leading dot, two decimals.
 * `.60`, and an em dash where a signal contributed nothing at all.
 */
export function formatWeight(value) {
  if (value === null || value === undefined) return '—'
  return value.toFixed(2).replace(/^0/, '')
}

/** Scores and contributions keep full precision; this is the ledger, not the headline. */
export function formatScore(value) {
  return value.toFixed(3)
}

/** The headline reads as a percentage. Precision lives in the ledger. */
export function formatPercent(value) {
  return `${Math.round(value * 100)}%`
}
