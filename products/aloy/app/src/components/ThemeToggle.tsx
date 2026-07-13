import { Moon, Sun } from 'lucide-react';
import { useTheme } from '@/hooks/useTheme';

/** Sidebar control that flips light/dark. Matches the rail buttons' styling and
 *  collapses to an icon on the slim rail. */
export function ThemeToggle({ expanded }: { expanded: boolean }) {
  const { theme, toggle } = useTheme();
  const isDark = theme === 'dark';
  const label = isDark ? 'Light mode' : 'Dark mode';

  return (
    <button
      type="button"
      onClick={toggle}
      title={label}
      aria-label={label}
      className={`flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium text-zinc-400 transition-colors hover:bg-zinc-800/70 hover:text-zinc-100 ${
        expanded ? '' : 'lg:justify-center lg:px-0'
      }`}
    >
      {isDark ? (
        <Sun size={18} className="text-zinc-500" />
      ) : (
        <Moon size={18} className="text-zinc-500" />
      )}
      <span className={expanded ? '' : 'lg:hidden'}>{label}</span>
    </button>
  );
}
