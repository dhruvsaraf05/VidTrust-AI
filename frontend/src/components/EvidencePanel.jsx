import { useState } from 'react'

import { explain } from '../api/client'

/**
 * The evidence behind each signal's reading, fetched on request.
 *
 * On request rather than automatically because the heatmap is a second
 * gradient pass through a 744 MB classifier -- seconds on CPU -- and most
 * viewers want the verdict, not the attribution.
 *
 * The heatmap NEVER appears without its measured agreement with occlusion
 * (saliency_faithfulness.py), and on this checkpoint it does not appear at
 * all: measured over 20 images the agreement is no better than chance, so the
 * backend's faithfulness gate declines and this shows its reason instead of
 * an overlay that would look like an explanation without being one.
 */
function Faithfulness({ summary }) {
  if (!summary || summary.rho_median == null) {
    return (
      <p className="mt-2 text-[0.8125rem] text-ochre">
        Agreement with occlusion: not measured on this machine.
      </p>
    )
  }
  return (
    <p className="mt-2 text-[0.8125rem] text-ink-3">
      Measured against occlusion over {summary.n} evaluation images: median
      Spearman <span className="fig">ρ = {summary.rho_median.toFixed(2)}</span>{' '}
      (1 is perfect agreement, 0 is none), positive on{' '}
      <span className="fig">{summary.rho_positive}</span> of{' '}
      <span className="fig">{summary.n}</span>. A map is drawn only above{' '}
      <span className="fig">ρ = {summary.min_faithful_rho.toFixed(1)}</span>.
    </p>
  )
}

function Heatmap({ saliency, imageUrl }) {
  return (
    <div>
      <h4 className="label">Classifier · where the score is attributed</h4>
      {saliency.available ? (
        <div className="relative mt-3 inline-block max-w-full">
          <img src={imageUrl} alt="The analysed image" className="block max-h-64 max-w-full" />
          <img
            src={`data:image/png;base64,${saliency.heatmap_png_b64}`}
            alt="Grad-CAM attribution overlaid on the image"
            className="absolute inset-0 h-full w-full"
          />
        </div>
      ) : (
        <p className="mt-3 text-sm text-ink-2">{saliency.detail}</p>
      )}
      <Faithfulness summary={saliency.faithfulness} />
    </div>
  )
}

function Spectrum({ spectrum }) {
  return (
    <div>
      <h4 className="label">Frequency · the spectrum the score reads</h4>
      {spectrum.available ? (
        <img
          src={`data:image/png;base64,${spectrum.spectrum_png_b64}`}
          alt="Log-magnitude Fourier spectrum with the low-frequency disc outlined"
          className="mt-3 block h-48 w-48"
        />
      ) : null}
      <p className="mt-2 text-[0.8125rem] text-ink-3">
        Low frequencies sit at the centre; the score is the share of energy
        outside the outlined disc. {spectrum.detail}
      </p>
    </div>
  )
}

function Provenance({ provenance }) {
  const { fields, raw_scan: raw, signal } = provenance
  return (
    <div>
      <h4 className="label">Provenance · every field the scan read</h4>
      <p className="mt-3 text-sm text-ink-2">{signal.detail}</p>
      {fields.length === 0 ? (
        <p className="mt-2 text-[0.8125rem] text-ink-3">
          No embedded metadata fields in this file.
        </p>
      ) : (
        <div className="mt-3 overflow-x-auto">
          <table className="w-full text-[0.8125rem]">
            <tbody>
              {fields.map((field, i) => (
                <tr key={`${field.key}-${i}`} className="border-t border-rule align-top">
                  <td className="py-1.5 pr-3 whitespace-nowrap text-ink-3">{field.source}</td>
                  <td className="fig py-1.5 pr-3 whitespace-nowrap">{field.key}</td>
                  <td className="fig py-1.5 break-all text-ink-2">
                    {field.value}
                    {field.truncated && <span className="text-ink-3"> …</span>}
                    {field.matched && (
                      <span className="ml-2 text-ochre">matched “{field.matched}”</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="mt-2 text-[0.8125rem] text-ink-3">
        Raw byte scan of the first and last{' '}
        <span className="fig">{Math.round(raw.bytes_each_end / 1024)}</span> KB:{' '}
        {raw.matched ? `matched “${raw.matched}”` : 'no generator marker'}.
      </p>
    </div>
  )
}

function readAsDataUrl(file) {
  return new Promise((resolve) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result)
    reader.onerror = () => resolve(null)
    reader.readAsDataURL(file)
  })
}

export default function EvidencePanel({ file }) {
  // Tagged with the file it belongs to: evidence for a previous file must
  // never sit beside a new verdict, so anything for another file reads idle.
  const [loaded, setLoaded] = useState({ file: null, status: 'idle' })
  const state = loaded.file === file ? loaded : { status: 'idle' }

  async function load() {
    setLoaded({ file, status: 'loading' })
    // A data URL rather than an object URL: nothing to revoke, so no effect
    // cleanup that StrictMode's double-invoke can fire before the image loads.
    const [outcome, imageUrl] = await Promise.all([explain(file), readAsDataUrl(file)])
    setLoaded(
      outcome.ok
        ? { file, status: 'done', data: outcome.data, imageUrl }
        : { file, status: 'failed', message: outcome.message },
    )
  }

  return (
    <section className="mt-12" aria-labelledby="evidence-heading">
      <h3 id="evidence-heading" className="label">
        Evidence behind each signal
      </h3>

      {state.status === 'idle' && (
        <button
          type="button"
          onClick={load}
          className="label mt-3 cursor-pointer rounded-[2px] border border-rule px-3 py-1.5 transition-colors hover:border-ink hover:text-ink"
        >
          Show evidence
        </button>
      )}
      {state.status === 'loading' && (
        <p className="label mt-3 animate-pulse">
          Running attribution through the classifier — a few seconds on CPU
        </p>
      )}
      {state.status === 'failed' && (
        <p className="panel mt-3 px-5 py-4 text-sm text-ink-2">{state.message}</p>
      )}
      {state.status === 'done' && (
        <div className="panel mt-3 space-y-8 px-5 py-5">
          <Heatmap saliency={state.data.saliency} imageUrl={state.imageUrl} />
          <Spectrum spectrum={state.data.spectrum} />
          <Provenance provenance={state.data.provenance} />
        </div>
      )}
    </section>
  )
}
