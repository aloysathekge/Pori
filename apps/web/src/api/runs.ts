import { apiFetch } from './client';
import type { RunResponse } from '@/types';

export function createRun(task: string, max_steps = 15) {
  return apiFetch<RunResponse>('/runs', {
    method: 'POST',
    body: JSON.stringify({ task, max_steps }),
  });
}

export function listRuns() {
  return apiFetch<RunResponse[]>('/runs');
}

export function getRun(id: string) {
  return apiFetch<RunResponse>(`/runs/${id}`);
}
