import type { ReactNode } from 'react';
import {
  SurfacePanel,
  SurfaceStatus,
  type SurfaceStatusKind,
} from '@aloy/surface';

/* Baseline primitives. These are the stable anchors every later revision
   extends: reuse them for new views instead of restyling from scratch. */

export function Section({
  heading,
  children,
}: {
  heading: string;
  children: ReactNode;
}) {
  return (
    <SurfacePanel className="baseline-section">
      <h2>{heading}</h2>
      {children}
    </SurfacePanel>
  );
}

export function EmptyState({
  status,
  message,
}: {
  status: SurfaceStatusKind;
  message: string;
}) {
  return <SurfaceStatus status={status === 'ready' ? 'empty' : status} message={message} />;
}

export function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="baseline-stat">
      <span className="baseline-stat-value">{value}</span>
      <span className="baseline-stat-label">{label}</span>
    </div>
  );
}
