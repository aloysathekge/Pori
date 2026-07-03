import { apiFetch } from './client';
import type { TraceListItem, TraceDetail } from '@/types';

export function listTraces(limit = 50, offset = 0) {
  return apiFetch<TraceListItem[]>(`/traces?limit=${limit}&offset=${offset}`);
}

export function getTrace(id: string) {
  return apiFetch<TraceDetail>(`/traces/${id}`);
}

export function deleteTrace(id: string) {
  return apiFetch<void>(`/traces/${id}`, { method: 'DELETE' });
}
