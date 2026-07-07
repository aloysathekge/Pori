import type { ReactNode } from 'react';

const colorMap: Record<string, string> = {
  green: 'bg-emerald-50 text-emerald-700 border-emerald-200',
  red: 'bg-red-900/50 text-red-700 border-red-700',
  yellow: 'bg-amber-50 text-amber-700 border-amber-200',
  blue: 'bg-blue-900/50 text-blue-300 border-blue-700',
  gray: 'bg-zinc-800 text-zinc-400 border-zinc-700',
  accent: 'bg-accent-50 text-accent-700 border-accent-200',
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
