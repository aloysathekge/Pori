import type { ReactNode } from 'react';
import {
  SurfacePanel,
  SurfaceStatus,
  useSurfaceResourceState,
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

/* Every view that presents data from a declared capability must render it
   inside ResourceSection (or spread the hook's feedbackProps itself): the
   host's quality gate requires a visible SDK-bound region for each declared
   resource so loading/empty/error truth stays visible to the user. */
export function ResourceSection({
  heading,
  resource,
  children,
}: {
  heading: string;
  resource: string;
  children: ReactNode;
}) {
  const state = useSurfaceResourceState(resource);
  return (
    <SurfacePanel className="baseline-section">
      <h2>{heading}</h2>
      <div {...state.feedbackProps}>
        {state.status !== 'ready' ? (
          <SurfaceStatus status={state.status} message={state.message} />
        ) : null}
        {children}
      </div>
    </SurfacePanel>
  );
}

export function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="baseline-stat">
      <span className="baseline-stat-value">{value}</span>
      <span className="baseline-stat-label">{label}</span>
    </div>
  );
}
