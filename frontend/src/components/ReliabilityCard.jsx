/**
 * How far the measured robustness study says to trust a verdict on THIS file.
 *
 * The backend reads the input's condition (format, estimated JPEG quality,
 * surviving EXIF, whether a platform re-encoded it), maps it to the nearest
 * condition it measured, and quotes what the system achieved there. Nothing
 * here is computed in the browser.
 *
 * It leads with detections at the deployed thresholds, not AUC: AUC barely
 * moves under recompression while the number of generated images actually
 * flagged falls sharply, and the user gets the thresholded behaviour. The
 * caveats are shown, not hidden behind a toggle -- they are what stop these
 * figures being read as a guarantee.
 */
function Row({ label, measured, clean, showClean }) {
  return (
    <tr className="border-t border-rule">
      <th scope="row" className="py-2 pr-4 text-left font-normal text-ink-2">
        {label}
      </th>
      <td className="fig py-2 pr-4 text-right">{measured}</td>
      {showClean && <td className="fig py-2 text-right text-ink-3">{clean}</td>}
    </tr>
  )
}

export default function ReliabilityCard({ card }) {
  if (!card) return null

  if (!card.available) {
    return (
      <section className="mt-12" aria-labelledby="reliability-heading">
        <h3 id="reliability-heading" className="label">
          Reliability in this condition
        </h3>
        <p className="panel mt-3 px-5 py-6 text-sm text-ink-2">{card.detail}</p>
      </section>
    )
  }

  const { measured, clean, condition } = card
  const showClean = condition !== 'clean'

  return (
    <section className="mt-12" aria-labelledby="reliability-heading">
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <h3 id="reliability-heading" className="label">
          Reliability in this condition
        </h3>
        <p className="label label-faint">nearest measured: {card.condition_label}</p>
      </div>

      <div className="panel mt-3 px-5 py-5">
        <p className="max-w-prose text-sm text-ink-2">
          {card.reason.charAt(0).toUpperCase() + card.reason.slice(1)}.
        </p>

        <table className="mt-4 w-full max-w-xl text-sm">
          <thead>
            <tr className="label label-faint">
              <th scope="col" className="pb-1 text-left font-normal">
                Measured on 400 public stills
              </th>
              <th scope="col" className="pb-1 pr-4 text-right font-normal">
                this condition
              </th>
              {showClean && (
                <th scope="col" className="pb-1 text-right font-normal">
                  clean originals
                </th>
              )}
            </tr>
          </thead>
          <tbody>
            <Row
              label="Generated images flagged"
              measured={`${measured.generated_flagged} / ${measured.generated_total}`}
              clean={`${clean.generated_flagged} / ${clean.generated_total}`}
              showClean={showClean}
            />
            <Row
              label="Real images flagged"
              measured={`${measured.real_flagged} / ${measured.real_total}`}
              clean={`${clean.real_flagged} / ${clean.real_total}`}
              showClean={showClean}
            />
            <Row
              label="Ranking quality (AUC)"
              measured={measured.auc_ensemble.toFixed(3)}
              clean={clean.auc_ensemble.toFixed(3)}
              showClean={showClean}
            />
          </tbody>
        </table>

        <ul className="mt-4 max-w-prose list-inside list-disc space-y-1 text-[0.8125rem] text-ink-3">
          {card.caveats.map((caveat) => (
            <li key={caveat}>{caveat}</li>
          ))}
        </ul>
      </div>
    </section>
  )
}
