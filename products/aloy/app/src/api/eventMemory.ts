import { apiFetch } from './client';

export type EventMemoryKind = 'semantic' | 'episodic' | 'procedural';
export type EventMemoryScope = 'event' | 'global';

export interface EventMemoryRecord {
  id: string;
  organization_id: string;
  user_id: string;
  event_id: string | null;
  agent_id: string | null;
  session_id: string | null;
  content: string;
  tags: string[] | null;
  importance: number;
  kind: EventMemoryKind;
  confidence: number;
  sensitivity: 'public' | 'internal' | 'confidential' | 'restricted';
  source: string;
  provenance: Record<string, unknown>;
  retention: Record<string, unknown>;
  conflict_key: string | null;
  status: string;
  superseded_by: string | null;
  created_at: string | null;
  updated_at: string | null;
  event_at: string | null;
  scope: EventMemoryScope;
  can_correct: boolean;
  can_forget: boolean;
  can_promote: boolean;
  promoted_global_id: string | null;
}

export interface EventMemoryResponse {
  event_id: string;
  event_records: EventMemoryRecord[];
  inherited_global_records: EventMemoryRecord[];
  event_count: number;
  inherited_global_count: number;
}

export interface EventMemoryWriteResponse {
  record: EventMemoryRecord;
  created: boolean;
}

export function getEventMemory(eventId: string) {
  return apiFetch<EventMemoryResponse>(`/events/${eventId}/memory`);
}

export function correctEventMemory(
  eventId: string,
  memoryId: string,
  content: string,
  reason?: string,
) {
  return apiFetch<EventMemoryWriteResponse>(
    `/events/${eventId}/memory/${memoryId}/corrections`,
    {
      method: 'POST',
      body: JSON.stringify({ content, reason: reason || null }),
    },
  );
}

export function forgetEventMemory(eventId: string, memoryId: string) {
  return apiFetch<void>(`/events/${eventId}/memory/${memoryId}`, {
    method: 'DELETE',
  });
}

export function promoteEventMemory(eventId: string, memoryId: string) {
  return apiFetch<EventMemoryWriteResponse>(
    `/events/${eventId}/memory/${memoryId}/promote`,
    { method: 'POST' },
  );
}
