import type { ReactNode } from 'react';

// Opacity-based tints read correctly on both the light and dark ground (a solid
// -50 tint would be a bright island in dark mode); gray/accent ride the themed
// zinc/accent variables.
const colorMap: Record<string, string> = {
  green: 'bg-emerald-500/12 text-emerald-600 border-emerald-500/30',
  red: 'bg-red-500/12 text-red-600 border-red-500/30',
  yellow: 'bg-amber-500/12 text-amber-600 border-amber-500/30',
  blue: 'bg-blue-500/12 text-blue-600 border-blue-500/30',
  gray: 'bg-zinc-800 text-zinc-400 border-zinc-700',
  accent: 'bg-accent-600/12 text-accent-700 border-accent-600/30',
};

export function Badge({
  children,
  color = 'gray',
}: {
  children: ReactNode;
  color?: keyof typeof colorMap;
}) {
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium ${colorMap[color] || colorMap.gray}`}
    >
      {children}
    </span>
  );
}
