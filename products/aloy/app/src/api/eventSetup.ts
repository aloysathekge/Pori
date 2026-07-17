import { apiFetch, apiUploadFile } from './client';
import type { EventSummary } from './events';
import type { ConnectionScope } from './connections';

export type EventSetupMode = 'simple' | 'assisted';

export interface EventSetupContextItem {
  id: string;
  event_id: string | null;
  kind: 'file' | 'link' | 'connection' | 'template';
  status: 'pending' | 'ingesting' | 'ready' | 'failed';
  label: string;
  source_url: string | null;
  content_type: string | null;
  size_bytes: number;
  sensitivity: string;
  attempt_count: number;
  max_attempts: number;
  next_attempt_at: string | null;
  retrieved_at: string | null;
  ingested_at: string | null;
  error: string | null;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface EventSetupDraft {
  id: string;
  mode: EventSetupMode;
  status: 'open' | 'promoted' | 'abandoned';
  title: string;
  description: string;
  created_event_id: string | null;
  context_items: EventSetupContextItem[];
  created_at: string;
  updated_at: string;
}

export function currentEventDraft() {
  return apiFetch<EventSetupDraft | undefined>('/event-drafts/current');
}

export function createEventDraft(input: Partial<Pick<EventSetupDraft, 'mode' | 'title' | 'description'>> = {}) {
  return apiFetch<EventSetupDraft>('/event-drafts', {
    method: 'POST',
    body: JSON.stringify(input),
  });
}

export function updateEventDraft(
  draftId: string,
  input: Partial<Pick<EventSetupDraft, 'mode' | 'title' | 'description'>>,
) {
  return apiFetch<EventSetupDraft>(`/event-drafts/${draftId}`, {
    method: 'PATCH',
    body: JSON.stringify(input),
  });
}

export function addEventDraftLink(draftId: string, url: string, label = '') {
  return apiFetch<EventSetupContextItem>(`/event-drafts/${draftId}/context`, {
    method: 'POST',
    body: JSON.stringify({ kind: 'link', url, label }),
  });
}

export function addEventDraftConnection(
  draftId: string,
  provider: string,
  connectionScope: ConnectionScope,
  label = '',
) {
  return apiFetch<EventSetupContextItem>(`/event-drafts/${draftId}/context`, {
    method: 'POST',
    body: JSON.stringify({
      kind: 'connection',
      provider,
      connection_scope: connectionScope,
      label,
      access_scope: { mode: 'event', resources: [] },
    }),
  });
}

export function uploadEventDraftFile(
  draftId: string,
  file: File,
  onProgress?: (progress: number) => void,
) {
  return apiUploadFile<EventSetupContextItem>(
    `/event-drafts/${draftId}/files`,
    file,
    onProgress,
  );
}

export function removeEventDraftContext(draftId: string, itemId: string) {
  return apiFetch<void>(`/event-drafts/${draftId}/context/${itemId}`, {
    method: 'DELETE',
  });
}

export function promoteEventDraft(draftId: string) {
  return apiFetch<EventSummary>(`/event-drafts/${draftId}/promote`, {
    method: 'POST',
  });
}

export function retryEventContext(eventId: string, itemId: string) {
  return apiFetch<EventSetupContextItem>(
    `/events/${eventId}/context/${itemId}/retry`,
    { method: 'POST' },
  );
}
