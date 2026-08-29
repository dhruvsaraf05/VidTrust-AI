import { MODES } from '../api/client'

/**
 * Mock/live switch, so the UI can be built with the backend stopped.
 *
 * Mock serves the captured fixtures in src/mocks/, which were generated from
 * real /api/analyze responses rather than hand-written -- so the shapes cannot
 * drift from the frozen contract.
 */
export default function ModeToggle({ mode, onChange, disabled }) {
  return (
    <div
      role="group"
      aria-label="Data source"
      className="flex items-center rounded-sm border border-rule bg-surface p-0.5"
    >
      {[
        { value: MODES.MOCK, label: 'Mock' },
        { value: MODES.LIVE, label: 'Live' },
      ].map((option) => {
        const active = mode === option.value
        return (
          <button
            key={option.value}
            type="button"
            disabled={disabled}
            aria-pressed={active}
            onClick={() => onChange(option.value)}
            className={[
              'cursor-pointer px-3 py-1 text-xs font-medium tracking-wide transition-colors disabled:cursor-not-allowed',
              active
                ? 'bg-ink text-white'
                : 'text-ink-muted hover:text-ink disabled:hover:text-ink-muted',
            ].join(' ')}
          >
            {option.label}
          </button>
        )
      })}
    </div>
  )
}
