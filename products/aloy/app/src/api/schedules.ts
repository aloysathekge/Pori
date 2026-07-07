import { apiFetch } from './client';

export interface ScheduleResponse {
  id: string;
  name: string;
  task: string;
  schedule: string;
  enabled: boolean;
  max_steps: number;
  conversation_id: string | null;
  next_run_at: string | null;
  last_run_at: string | null;
  last_run_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface ScheduleCreate {
  name: string;
  task: string;
  schedule: string;
  max_steps?: number;
  conversation_id?: string | null;
}

export type ScheduleUpdate = Partial<ScheduleCreate> & { enabled?: boolean };

export function listSchedules() {
  return apiFetch<ScheduleResponse[]>('/cron');
}

export function createSchedule(data: ScheduleCreate) {
  return apiFetch<ScheduleResponse>('/cron', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export function updateSchedule(id: string, data: ScheduleUpdate) {
  return apiFetch<ScheduleResponse>(`/cron/${id}`, {
    method: 'PATCH',
    body: JSON.stringify(data),
  });
}

export function deleteSchedule(id: string) {
  return apiFetch<void>(`/cron/${id}`, { method: 'DELETE' });
}
