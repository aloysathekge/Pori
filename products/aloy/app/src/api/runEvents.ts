import { apiFetch } from './client';
import type { RunTimelinePage } from '@/types';

export interface RunEvent {
  type: string;
  payload: Record<string, unknown>;
  step: number;
}

export interface RunEventLog {
  run_id: string;
  conversation_id: string | null;
  events: RunEvent[];
  event_count: number;
  created_at: string;
}

export function getRunEvents(runId: string) {
  return apiFetch<RunEventLog>(`/runs/${runId}/events`);
}

export function getRunTimeline(runId: string, after = 0) {
  return apiFetch<RunTimelinePage>(`/runs/${runId}/timeline?after=${after}`);
}
