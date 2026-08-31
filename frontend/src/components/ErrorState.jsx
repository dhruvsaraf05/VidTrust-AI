/**
 * Failure replaces the result area entirely — it never sits above a stale
 * verdict from a previous file.
 *
 * Each code gets a plain statement of what happened and one next action. No
 * apologies, no vagueness about the cause.
 *
 * There is deliberately NO state for NO_FACES_DETECTED. That code exists only
 * so the frozen contract is honoured; nothing in this pipeline inspects faces
 * and no code path can raise it. Building a screen for it would put a dead
 * state in the demo. If it somehow arrives, it falls through to the default.
 */
const MESSAGES = {
  UNSUPPORTED_FORMAT: {
    title: 'That file type is not supported',
    action: 'Convert it first, or choose a different file.',
  },
  FILE_TOO_LARGE: {
    title: 'That file is over the 50 MB limit',
    action: 'Trim the clip or export it at a lower resolution, then try again.',
  },
  PROCESSING_FAILED: {
    title: 'The file could not be read',
    action: 'Re-export it from the original source, then try again.',
  },
  MODEL_UNAVAILABLE: {
    title: 'The classifier is not loaded',
    action:
      'Restart the backend and check its startup log for the model download.',
  },
  NETWORK_ERROR: {
    title: 'No response from the backend',
    action: 'Start the server, then upload the file again.',
    command: 'cd backend && .venv/Scripts/python -m uvicorn main:app --reload',
  },
}

const FALLBACK = {
  title: 'The analysis did not complete',
  action: 'Try the file again, or check the backend log for details.',
}

export default function ErrorState({ code, message }) {
  const copy = MESSAGES[code] ?? FALLBACK

  return (
    <section
      className="mt-10 border-l-2 border-crimson bg-crimson-tint px-6 py-6"
      role="alert"
      aria-labelledby="error-heading"
    >
      <p className="fig text-[0.8125rem] tracking-[0.08em] text-crimson">
        {code}
      </p>

      <h2
        id="error-heading"
        className="mt-2 text-xl font-semibold text-ink"
      >
        {copy.title}
      </h2>

      {message && <p className="mt-2 max-w-prose text-ink-2">{message}</p>}

      <p className="mt-4 max-w-prose text-ink">{copy.action}</p>

      {copy.command && (
        <pre className="fig mt-3 overflow-x-auto rounded-[3px] border border-rule bg-panel px-3 py-2 text-[0.8125rem] text-ink-2">
          {copy.command}
        </pre>
      )}
    </section>
  )
}
