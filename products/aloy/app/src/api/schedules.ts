import { apiFetch } from './client';

export interface ScheduleResponse {
  id: string;
  event_id: string | null;
  name: string;
  task: string;
  schedule: string;
  timezone: string;
  authority: 'report_only' | 'organize';
  notification_mode: 'attention' | 'always';
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
  event_id: string;
  name: string;
  task: string;
  schedule: string;
  timezone: string;
  authority: 'report_only' | 'organize';
  notification_mode: 'attention' | 'always';
  max_steps?: number;
}

export type ScheduleUpdate = Partial<Omit<ScheduleCreate, 'event_id'>> & { enabled?: boolean };

export interface ScheduleRunResponse {
  id: string;
  status: string;
  success: boolean;
  final_answer: string | null;
  steps_taken: number;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

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

export function listScheduleRuns(id: string, limit = 8) {
  return apiFetch<ScheduleRunResponse[]>(`/cron/${id}/runs?limit=${limit}`);
}
