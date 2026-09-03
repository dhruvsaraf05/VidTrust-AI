import { useEffect, useState } from 'react'
import { Moon, Sun } from 'lucide-react'

const KEY = 'vidtrust.theme'

/**
 * Light / dark toggle.
 *
 * First visit: follows the OS preference (prefers-color-scheme), so the tool
 * matches the room it's being demoed in without asking. Once the button is
 * used, that explicit choice is remembered and the OS preference is no longer
 * consulted -- a person who deliberately picked dark should not have it
 * reverted by their system switching at sunset.
 */
function initialTheme() {
  const stored = localStorage.getItem(KEY)
  if (stored === 'light' || stored === 'dark') return stored
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches
    ? 'dark'
    : 'light'
}

export default function ThemeToggle() {
  const [theme, setTheme] = useState(initialTheme)

  useEffect(() => {
    if (theme === 'dark') {
      document.documentElement.dataset.theme = 'dark'
    } else {
      delete document.documentElement.dataset.theme
    }
    localStorage.setItem(KEY, theme)
  }, [theme])

  const dark = theme === 'dark'

  return (
    <button
      type="button"
      onClick={() => setTheme(dark ? 'light' : 'dark')}
      aria-label={dark ? 'Switch to light mode' : 'Switch to dark mode'}
      title={dark ? 'Switch to light mode' : 'Switch to dark mode'}
      className="flex cursor-pointer items-center justify-center rounded-[3px] border border-rule bg-panel p-1.5 text-ink-2 transition-colors hover:text-ink"
    >
      {dark ? (
        <Sun className="h-4 w-4" strokeWidth={1.5} />
      ) : (
        <Moon className="h-4 w-4" strokeWidth={1.5} />
      )}
    </button>
  )
}
