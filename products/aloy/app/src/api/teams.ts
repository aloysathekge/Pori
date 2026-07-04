import { apiFetch } from './client';
import type { TeamConfigCreate, TeamConfigResponse, TeamRunResponse } from '@/types';

export function listTeams() {
  return apiFetch<TeamConfigResponse[]>('/teams');
}

export function getTeam(id: string) {
  return apiFetch<TeamConfigResponse>(`/teams/${id}`);
}

export function createTeam(data: TeamConfigCreate) {
  return apiFetch<TeamConfigResponse>('/teams', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export function updateTeam(id: string, data: Partial<TeamConfigCreate>) {
  return apiFetch<TeamConfigResponse>(`/teams/${id}`, {
    method: 'PATCH',
    body: JSON.stringify(data),
  });
}

export function deleteTeam(id: string) {
  return apiFetch<void>(`/teams/${id}`, { method: 'DELETE' });
}

export function runTeam(id: string, task: string) {
  return apiFetch<TeamRunResponse>(`/teams/${id}/run`, {
    method: 'POST',
    body: JSON.stringify({ task }),
  });
}
