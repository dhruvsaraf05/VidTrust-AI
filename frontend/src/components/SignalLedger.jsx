import {
  buildLedger,
  displayContributions,
  formatScore,
  formatWeight,
  unavailableRows,
} from '../lib/fusion'

const GRID =
  'grid grid-cols-[1fr_auto] sm:grid-cols-[8.5rem_minmax(0,1fr)_7rem_6rem] items-center gap-x-5'

/**
 * The signal ledger — the page's centrepiece.
 *
 * A verdict here is a weighted sum, so this is laid out as a worked
 * calculation rather than a table of readings: each row shows its score, the
 * weight actually applied to it, and what that multiplies out to, and the
 * contributions total under a rule to the fused confidence.
 *
 * The weight column is the important one. It reads "nominal ▸ effective". When
 * every signal ran those are identical and the column is quiet. When one could
 * not run, its effective weight becomes an em dash and the others visibly
 * WIDEN to absorb it. That is the renormalisation invariant made legible: a
 * missing signal is removed from the average, never counted as a zero that
 * would drag the verdict toward "real".
 */
export default function SignalLedger({ signals, confidence }) {
  const rows = buildLedger(signals)
  // Displayed contributions are reconciled so the column sums to the total.
  const shown = displayContributions(rows, confidence)
  const missing = unavailableRows(rows)
  const widened = rows.filter((row) => row.widened)
  const allMissing = missing.length === rows.length

  return (
    <section className="mt-12" aria-labelledby="ledger-heading">
      <div className="flex items-baseline justify-between gap-4">
        <h3 id="ledger-heading" className="label">
          Signal ledger
        </h3>
        <p className="label label-faint hidden sm:block">
          score × effective weight
        </p>
      </div>

      <div className="panel mt-3 px-5 py-1">
        {/* Column headings, desktop only — on narrow screens each figure
            carries its own label instead. */}
        <div className={`${GRID} hidden border-b border-rule py-3 sm:grid`}>
          <span className="label label-faint">Signal</span>
          <span className="label label-faint">Reading</span>
          <span className="label label-faint text-right">Weight</span>
          <span className="label label-faint text-right">Contribution</span>
        </div>

        {rows.map((row) => (
          <LedgerRow key={row.key} row={row} shown={shown.get(row.key)} />
        ))}

        {/* The rule and the total. The column above it sums to this figure. */}
        <div className={`${GRID} border-t-2 border-ink py-4`}>
          <span className="text-sm font-semibold sm:col-span-3 sm:text-right">
            Fused confidence
          </span>
          <span className="fig text-right text-lg font-semibold">
            {formatScore(confidence)}
          </span>
        </div>
      </div>

      {allMissing ? (
        <Note>
          No signal could run on this file. Confidence is held at{' '}
          <span className="fig">0.500</span> — the midpoint — rather than
          defaulting to a reading in either direction.
        </Note>
      ) : missing.length > 0 ? (
        <Note>
          {joinNames(missing)} could not run.{' '}
          {missing.length === 1 ? 'Its' : 'Their'}{' '}
          <span className="fig">{formatWeight(missingWeight(missing))}</span> is
          redistributed across the remaining signals, not counted as zero —{' '}
          {widened.map((row, index) => (
            <span key={row.key}>
              {index > 0 && index === widened.length - 1 ? ' and ' : index > 0 ? ', ' : ''}
              {row.name.toLowerCase()} now carries{' '}
              <span className="fig font-medium">
                {formatWeight(row.effectiveWeight)}
              </span>
            </span>
          ))}
          .
        </Note>
      ) : null}
    </section>
  )
}

function LedgerRow({ row, shown }) {
  return (
    <div className="border-b border-rule py-4 last:border-b-0">
      <div className={GRID}>
        <span className="col-span-2 text-sm font-medium sm:col-span-1">
          {row.name}
        </span>

        <span className="col-span-2 mt-2 sm:col-span-1 sm:mt-0">
          <ScoreBar row={row} />
        </span>

        <span className="mt-3 block sm:mt-0 sm:text-right">
          <span className="label label-faint mb-1 block sm:hidden">Weight</span>
          <WeightCell row={row} />
        </span>

        <span className="mt-3 block text-right sm:mt-0">
          <span className="label label-faint mb-1 block sm:hidden">
            Contribution
          </span>
          <span
            className={`fig text-sm ${row.available ? '' : 'text-ink-3'}`}
          >
            {row.available ? formatScore(shown ?? row.contribution) : '—'}
          </span>
        </span>
      </div>

      {/* Free text from the API, for display only — never parsed. */}
      <p className="mt-2.5 text-[0.8125rem] leading-snug text-ink-3">
        {row.detail}
      </p>
    </div>
  )
}

/**
 * A signal that could not run gets a hatched "no reading" track, not an empty
 * bar. An empty bar on the same scale would read as a score of zero, which is
 * exactly the wrong conclusion — absence of evidence is not evidence of
 * authenticity.
 */
function ScoreBar({ row }) {
  if (!row.available) {
    return (
      <span className="flex h-6 items-center">
        <span className="hatch relative flex h-6 w-full items-center justify-center rounded-[2px] border border-rule bg-panel-sunk">
          <span className="label label-faint bg-panel px-1.5">Unavailable</span>
        </span>
      </span>
    )
  }

  return (
    <span className="flex h-6 items-center gap-3">
      <span className="h-2 flex-1 rounded-[2px] bg-panel-sunk">
        <span
          className="block h-2 rounded-[2px] bg-ink-2"
          style={{ width: `${Math.min(Math.max(row.score, 0), 1) * 100}%` }}
        />
      </span>
      <span className="fig w-12 shrink-0 text-right text-sm">
        {formatScore(row.score)}
      </span>
    </span>
  )
}

function WeightCell({ row }) {
  return (
    <span className="fig inline-flex items-baseline gap-1.5 text-sm">
      <span className="text-ink-3">{formatWeight(row.nominalWeight)}</span>
      <span aria-hidden="true" className="text-ink-3">
        ▸
      </span>
      {row.available ? (
        <span
          className={
            row.widened
              ? 'rounded-[2px] bg-accent-tint px-1.5 py-0.5 font-semibold text-accent'
              : ''
          }
        >
          {formatWeight(row.effectiveWeight)}
        </span>
      ) : (
        <span className="text-ink-3">—</span>
      )}
    </span>
  )
}

function Note({ children }) {
  return (
    <p className="mt-3 border-l-2 border-accent bg-accent-tint/50 px-4 py-3 text-sm text-ink-2">
      {children}
    </p>
  )
}

function joinNames(rows) {
  const names = rows.map((row) => row.name)
  if (names.length === 1) return names[0]
  return `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`
}

function missingWeight(rows) {
  return rows.reduce((sum, row) => sum + row.nominalWeight, 0)
}
