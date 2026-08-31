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
      className="flex items-center rounded-[3px] border border-rule bg-panel p-0.5"
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
              'label cursor-pointer rounded-[2px] px-3 py-1.5 transition-colors disabled:cursor-not-allowed',
              active
                ? 'bg-ink text-panel'
                : 'hover:text-ink disabled:hover:text-ink-2',
            ].join(' ')}
          >
            {option.label}
          </button>
        )
      })}
    </div>
  )
}
