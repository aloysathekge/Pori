import { apiFetch, apiTextFetch } from './client';

export interface SurfaceBuild {
  id: string;
  event_id: string;
  revision_id: string;
  status: 'pending' | 'running' | 'succeeded' | 'failed' | 'blocked' | string;
  bundle_available: boolean;
  bundle_sha256: string | null;
  bundle_size_bytes: number;
  diagnostics: Array<{ code?: string; message?: string }>;
  created_at: string;
  completed_at: string | null;
}

export function listSurfaceBuilds(eventId: string) {
  return apiFetch<SurfaceBuild[]>(`/events/${eventId}/surface/builds`);
}

export function getSurfaceRuntimeDocument(eventId: string, buildId: string) {
  return apiTextFetch(
    `/events/${eventId}/surface/builds/${buildId}/runtime-document`,
  );
}

export function surfaceSeenKey(eventId: string) {
  return `aloy:event:${eventId}:surface-seen-build`;
}
