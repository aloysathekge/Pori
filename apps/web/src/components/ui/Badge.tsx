import type { ReactNode } from 'react';

const colorMap: Record<string, string> = {
  green: 'bg-emerald-900/50 text-emerald-300 border-emerald-700',
  red: 'bg-red-900/50 text-red-300 border-red-700',
  yellow: 'bg-amber-900/50 text-amber-300 border-amber-700',
  blue: 'bg-blue-900/50 text-blue-300 border-blue-700',
  gray: 'bg-zinc-800 text-zinc-400 border-zinc-700',
  indigo: 'bg-indigo-900/50 text-indigo-300 border-indigo-700',
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
