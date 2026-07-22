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

export interface SurfaceActivity {
  run_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | 'overdue' | string;
  stage:
    | 'queued'
    | 'generating_candidate'
    | 'validating_candidate'
    | 'building_bundle'
    | 'inspecting_preview'
    | 'publishing_surface'
    | 'repairing_candidate'
    | string;
  message: string;
  submission: number;
  attempt_count: number;
  max_attempts: number;
  started_at: string;
  updated_at: string;
  completed_at: string | null;
  elapsed_seconds: number;
  active: boolean;
}

export interface SurfaceRuntimeContext {
  protocol_version: '1';
  command_contract_version: '1';
  sdk_version: '1';
  resource_state_version?: '1';
  event_id: string;
  project_id: string;
  build_id: string;
  code_revision_id: string;
  data_revision: number;
  capabilities: string[];
  widgets: string[];
  resource_states?: Record<string, {
    status:
      | 'loading'
      | 'ready'
      | 'empty'
      | 'stale'
      | 'error'
      | 'permission_denied'
      | 'pending'
      | 'indeterminate';
    message?: string;
    retryable: boolean;
  }>;
  data: {
    files?: SurfaceRuntimeFile[];
    [key: string]: unknown;
  };
}

export interface SurfaceRuntimeFile {
  id: string;
  name: string;
  kind: string;
  in_library?: boolean;
  content_type: string;
  size_bytes: number;
  origin_session_id: string | null;
  origin_run_id: string | null;
  created_at: string;
}

export type SurfaceInteractionMethod =
  | 'command'
  | 'dispatch'
  | 'ask_aloy'
  | 'request_action';

export interface SurfaceInteractionRequest {
  build_id: string;
  code_revision_id: string;
  data_revision: number;
  method: SurfaceInteractionMethod;
  name: string;
  component_id: string;
  payload: Record<string, unknown>;
  message?: string;
  reason?: string;
  resource_refs?: SurfaceResourceRef[];
  idempotency_key: string;
}

export interface SurfaceResourceRef {
  type: 'file';
  id: string;
}

export interface SurfaceInteractionResponse {
  id: string;
  status: string;
  name: string;
  interaction_class: string;
  data_revision: number | null;
  handling_run_id: string | null;
  proposal_id: string | null;
  request_message_id: string | null;
  outcome_message_id: string | null;
  result: Record<string, unknown>;
  replayed: boolean;
}

export interface PublishedSurfaceRuntime {
  project_id: string | null;
  published_revision_id: string | null;
  published_build_id: string | null;
  build: SurfaceBuild | null;
}

export interface SurfacePublication {
  id: string;
  event_id: string;
  project_id: string;
  revision_id: string;
  revision_number: number;
  build_id: string;
  previous_revision_id: string | null;
  previous_build_id: string | null;
  action: 'publish' | 'rollback';
  actor_id: string;
  run_id: string | null;
  created_at: string;
  replayed: boolean;
}

export interface SurfaceEvolutionProposal {
  id: string;
  event_id: string;
  project_id: string;
  trigger:
    | 'event_phase_changed'
    | 'capability_gap'
    | 'negative_feedback'
    | 'primary_job_failure'
    | 'quality_failure'
    | string;
  goal: string;
  status: 'pending' | 'dismissed' | 'queued' | 'superseded' | string;
  occurrence_count: number;
  reason_codes: string[];
  evidence_refs: string[];
  base_revision_id: string | null;
  base_build_id: string | null;
  base_data_revision: number;
  builder_run_id: string | null;
  decided_at: string | null;
  cooldown_until: string | null;
  created_at: string;
  updated_at: string;
}

export function listSurfaceBuilds(eventId: string) {
  return apiFetch<SurfaceBuild[]>(`/events/${eventId}/surface/builds`);
}

export function getSurfaceActivity(eventId: string) {
  return apiFetch<SurfaceActivity | null>(`/events/${eventId}/surface/status`);
}

export function getPublishedSurfaceRuntime(eventId: string) {
  return apiFetch<PublishedSurfaceRuntime>(`/events/${eventId}/surface/runtime`);
}

export function listSurfaceEvolutionProposals(eventId: string) {
  return apiFetch<SurfaceEvolutionProposal[]>(
    `/events/${eventId}/surface/evolution-proposals`,
  );
}

export function decideSurfaceEvolutionProposal(
  eventId: string,
  proposalId: string,
  decision: 'accept' | 'dismiss',
) {
  return apiFetch<SurfaceEvolutionProposal>(
    `/events/${eventId}/surface/evolution-proposals/${proposalId}/decision`,
    {
      method: 'POST',
      body: JSON.stringify({ decision }),
    },
  );
}

export function submitSurfaceFeedback(eventId: string, message?: string) {
  return apiFetch<SurfaceEvolutionProposal>(`/events/${eventId}/surface/feedback`, {
    method: 'POST',
    body: JSON.stringify(message ? { message } : {}),
  });
}

export function listSurfacePublications(eventId: string) {
  return apiFetch<SurfacePublication[]>(
    `/events/${eventId}/surface/publications`,
  );
}

export function rollbackSurface(
  eventId: string,
  request: {
    build_id: string;
    expected_published_revision_id: string | null;
    expected_published_build_id: string | null;
    idempotency_key: string;
  },
) {
  return apiFetch<SurfacePublication>(`/events/${eventId}/surface/rollback`, {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

export function getSurfaceRuntimeDocument(eventId: string, buildId: string) {
  return apiTextFetch(
    `/events/${eventId}/surface/builds/${buildId}/runtime-document`,
  );
}

export function getSurfaceRuntimeContext(
  eventId: string,
  buildId: string,
  signal?: AbortSignal,
) {
  return apiFetch<SurfaceRuntimeContext>(
    `/events/${eventId}/surface/context?build_id=${encodeURIComponent(buildId)}`,
    { signal },
  );
}

export function createSurfaceInteraction(
  eventId: string,
  request: SurfaceInteractionRequest,
  signal?: AbortSignal,
) {
  return apiFetch<SurfaceInteractionResponse>(
    `/events/${eventId}/surface/interactions`,
    { method: 'POST', body: JSON.stringify(request), signal },
  );
}

export function surfaceSeenKey(eventId: string) {
  return `aloy:event:${eventId}:surface-seen-build`;
}
