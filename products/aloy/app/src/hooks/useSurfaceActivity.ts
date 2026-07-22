import { useCallback, useEffect, useState } from 'react';
import { getSurfaceActivity, type SurfaceActivity } from '@/api/surfaces';
import { getEventSurface } from '@/api/events';

const ACTIVE_POLL_MS = 3_000;

async function activityFromTrail(eventId: string): Promise<SurfaceActivity | null> {
  const event = await getEventSurface(eventId);
  const activitySection = event.surface.sections.find((section) => section.kind === 'activity');
  if (!activitySection || activitySection.kind !== 'activity') return null;
  const entry = activitySection.entries.find((item) => [
    'surface_build_queued',
    'surface_build_started',
    'surface_build_retry_scheduled',
    'surface_build_failed',
    'surface_published',
  ].includes(item.kind));
  if (!entry || !entry.run_id) return null;
  const failed = entry.kind === 'surface_build_failed';
  const completed = entry.kind === 'surface_published';
  const queued = entry.kind === 'surface_build_queued';
  const retrying = entry.kind === 'surface_build_retry_scheduled';
  const materializing = entry.payload.mode === 'persisted_source';
  const startedAt = new Date(entry.created_at);
  const elapsed = Number.isNaN(startedAt.getTime())
    ? 0
    : Math.max(0, Math.floor((Date.now() - startedAt.getTime()) / 1_000));
  return {
    run_id: entry.run_id,
    status: failed ? 'failed' : completed ? 'completed' : queued ? 'pending' : 'running',
    stage: queued ? 'queued' : 'generating_candidate',
    message: failed
      ? 'The Surface could not be completed'
      : completed
        ? 'Your Surface is ready'
        : retrying
          ? materializing ? 'Retrying the starting Surface safely' : 'Retrying the Surface safely'
          : queued
            ? materializing ? 'Preparing your starting Surface' : 'Waiting for the Surface Builder'
            : materializing ? 'Checking and publishing your starting Surface' : 'Designing and writing your Surface',
    submission: 1,
    max_submissions: materializing ? 1 : 3,
    candidate_mode: null,
    generation_phase: null,
    output_chars: 0,
    output_chunks: 0,
    attempt_count: Number(entry.payload.attempt ?? 1),
    max_attempts: Number(entry.payload.max_attempts ?? 3),
    started_at: entry.created_at,
    updated_at: entry.created_at,
    completed_at: failed || completed ? entry.created_at : null,
    elapsed_seconds: elapsed,
    active: !failed && !completed,
  };
}

export function useSurfaceActivity(
  eventId: string,
  refreshKey?: string,
  enabled = true,
) {
  const [activity, setActivity] = useState<SurfaceActivity | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    if (!enabled) return null;
    try {
      let next: SurfaceActivity | null;
      try {
        next = await getSurfaceActivity(eventId);
      } catch {
        // During a rolling local update, the Trail already exposes coarse
        // lifecycle state even before the richer status route is restarted.
        next = await activityFromTrail(eventId);
      }
      setActivity(next);
      setError(null);
      return next;
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
      return null;
    }
  }, [enabled, eventId]);

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    async function poll() {
      const next = await refresh();
      if (cancelled) return;
      if (next?.active) timer = setTimeout(poll, ACTIVE_POLL_MS);
    }

    void poll();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [enabled, refresh, refreshKey]);

  return { activity, error, refresh };
}
