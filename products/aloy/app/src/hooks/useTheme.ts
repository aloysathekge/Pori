import { useState } from 'react';

/**
 * Light/dark theme, backed by the `data-theme` attribute on <html>. The initial
 * value is set pre-paint by the inline script in index.html (stored choice, else
 * OS preference), so first render already reads the resolved theme — no flash.
 * Setting it flips the attribute (which restyles every zinc-* utility via CSS
 * variables) and persists the choice.
 */

export type Theme = 'light' | 'dark';

const STORAGE_KEY = 'aloy.theme';

function currentTheme(): Theme {
  return document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light';
}

export function useTheme() {
  const [theme, setThemeState] = useState<Theme>(currentTheme);

  const setTheme = (next: Theme) => {
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // private mode / storage disabled — the attribute still applies for this session
    }
    setThemeState(next);
  };

  const toggle = () => setTheme(theme === 'dark' ? 'light' : 'dark');

  return { theme, setTheme, toggle };
}
