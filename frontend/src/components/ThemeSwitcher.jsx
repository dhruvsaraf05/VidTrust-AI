import { useEffect, useState } from 'react'

/**
 * TEMPORARY — palette comparison only. Delete once a direction is chosen,
 * along with themes.css and its import in main.jsx.
 *
 * Sets data-theme on <html>; the overrides in themes.css do the rest. Nothing
 * structural is themed, so switching changes colour and nothing else.
 */
const THEMES = [
  { value: '', label: 'Warm', title: 'Current: warm off-white, slate-indigo' },
  { value: 'dark', label: 'A · Dark', title: 'A: dark instrument' },
  { value: 'clinical', label: 'B · Clinical', title: 'B: cool clinical' },
  { value: 'greycard', label: 'C · Grey card', title: 'C: photographic neutral grey' },
]

const KEY = 'vidtrust.theme'

export default function ThemeSwitcher() {
  const [theme, setTheme] = useState(() => localStorage.getItem(KEY) ?? '')

  useEffect(() => {
    if (theme) {
      document.documentElement.dataset.theme = theme
    } else {
      delete document.documentElement.dataset.theme
    }
    localStorage.setItem(KEY, theme)
  }, [theme])

  return (
    <div
      role="group"
      aria-label="Palette (temporary)"
      className="flex items-center rounded-[3px] border border-rule bg-panel p-0.5"
    >
      {THEMES.map((option) => {
        const active = theme === option.value
        return (
          <button
            key={option.value || 'default'}
            type="button"
            title={option.title}
            aria-pressed={active}
            onClick={() => setTheme(option.value)}
            className={[
              'label cursor-pointer rounded-[2px] px-2.5 py-1.5 transition-colors',
              active ? 'bg-ink text-panel' : 'hover:text-ink',
            ].join(' ')}
          >
            {option.label}
          </button>
        )
      })}
    </div>
  )
}
