/**
 * The unedited API response, collapsed.
 *
 * Kept because in a demo it reads as technical honesty: every figure on the
 * page above can be traced back to a field here, and nothing is being
 * dramatised between the API and the screen.
 */
export default function RawResponse({ result }) {
  return (
    <details className="mt-12 border-t border-rule pt-4">
      <summary className="label cursor-pointer select-none hover:text-ink">
        Raw response
      </summary>
      <pre className="fig mt-3 max-h-96 overflow-auto rounded-[3px] border border-rule bg-panel p-4 text-[0.8125rem] leading-relaxed text-ink-2">
        {JSON.stringify(result, null, 2)}
      </pre>
    </details>
  )
}
