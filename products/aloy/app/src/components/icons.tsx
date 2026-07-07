/* Aloy icon family — hand-drawn for this product, not a stock set.
   Shared grammar: 24px grid, 1.75 rounded strokes, generous negative space,
   and one small filled "signal dot" per glyph (the agent's presence). Icons
   inherit currentColor so nav states/tints keep working. */

import type { SVGProps } from 'react';

type IconProps = SVGProps<SVGSVGElement> & { size?: number };

function base({ size = 20, ...props }: IconProps) {
  return {
    width: size,
    height: size,
    viewBox: '0 0 24 24',
    fill: 'none',
    stroke: 'currentColor',
    strokeWidth: 1.75,
    strokeLinecap: 'round' as const,
    strokeLinejoin: 'round' as const,
    'aria-hidden': true,
    ...props,
  };
}

/** Brand mark: the Aloy "A" in its rounded tile. */
export function AloyMark({ size = 28, ...props }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden {...props}>
      <rect width="32" height="32" rx="8" fill="#0F8571" />
      <path d="M16 7.5 L24 24 H19.6 L16 16.2 L12.4 24 H8 Z" fill="#fff" />
    </svg>
  );
}

/** Chat: a soft speech shape, the dot mid-thought. */
export function ChatIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M4.5 12a7.5 7.5 0 1 1 3.2 6.15L4 19l.9-3.4A7.4 7.4 0 0 1 4.5 12Z" />
      <path d="M8.8 12h.01M15.2 12h.01" />
      <circle cx="12" cy="12" r="1.4" fill="currentColor" stroke="none" />
    </svg>
  );
}

/** Agents: an orbiting presence around a core. */
export function AgentsIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <circle cx="12" cy="12" r="3.6" />
      <path d="M18.9 7.8a9 9 0 0 1 .0 8.4M5.1 7.8a9 9 0 0 0 0 8.4" />
      <circle cx="12" cy="4.4" r="1.4" fill="currentColor" stroke="none" />
    </svg>
  );
}

/** Skills: stacked capability layers, top one lit. */
export function SkillsIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="m12 4 8 4-8 4-8-4 8-4Z" />
      <path d="m5.2 11.4 6.8 3.4 6.8-3.4" />
      <path d="m5.2 15 6.8 3.4L18.8 15" />
      <circle cx="12" cy="8" r="1.4" fill="currentColor" stroke="none" />
    </svg>
  );
}

/** Schedules: time with the next fire marked. */
export function SchedulesIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <circle cx="12" cy="12" r="8" />
      <path d="M12 7.6V12l3 1.8" />
      <circle cx="17.6" cy="6.4" r="1.6" fill="currentColor" stroke="none" />
    </svg>
  );
}

/** Teams: three voices around a shared center. */
export function TeamsIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <circle cx="12" cy="6.4" r="2.4" />
      <circle cx="6" cy="16.6" r="2.4" />
      <circle cx="18" cy="16.6" r="2.4" />
      <path d="M10.6 8.6 7.2 14.4M13.4 8.6l3.4 5.8M8.4 16.6h7.2" />
      <circle cx="12" cy="13.2" r="1.2" fill="currentColor" stroke="none" />
    </svg>
  );
}

/** Memory: what's kept — a vessel with a living spark. */
export function MemoryIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M6 4.5h9.5a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H6a1.5 1.5 0 0 1-1.5-1.5v-12A1.5 1.5 0 0 1 6 4.5Z" />
      <path d="M8 4.5v15" />
      <path d="M12.8 8.4c1.6 0 2.6 1 2.6 2.3 0 1.6-1.5 1.9-1.5 3.2" />
      <circle cx="13.9" cy="16.4" r="1.2" fill="currentColor" stroke="none" />
    </svg>
  );
}

/** Usage: consumption bars, the live one marked. */
export function UsageIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M4.5 19.5v-6M10 19.5V9M15.5 19.5v-3.6M21 19.5V4.5" />
      <circle cx="15.5" cy="12.4" r="1.4" fill="currentColor" stroke="none" />
    </svg>
  );
}

/** Traces: the path a run actually took. */
export function TracesIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M4 17.5c3.5 0 3-11 6.5-11s3.2 7.5 6 7.5c1.6 0 2.3-1.4 3.5-1.4" />
      <circle cx="4" cy="17.5" r="1.4" fill="currentColor" stroke="none" />
      <circle cx="20" cy="12.6" r="1.4" fill="currentColor" stroke="none" />
    </svg>
  );
}

/** Settings: tuning rails, not a gear. */
export function SettingsIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M5 7.5h14M5 12h14M5 16.5h14" />
      <circle cx="9.2" cy="7.5" r="1.9" fill="var(--color-zinc-900, #ffffff)" />
      <circle cx="15" cy="12" r="1.9" fill="var(--color-zinc-900, #ffffff)" />
      <circle cx="7.6" cy="16.5" r="1.9" fill="var(--color-zinc-900, #ffffff)" />
    </svg>
  );
}
