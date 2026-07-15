import { apiFetch } from './client';

export interface EventSummary {
  id: string;
  type: 'life' | 'project' | string;
  title: string;
  lifecycle: string;
  phase: string;
  summary: string;
  is_life: boolean;
  conversation_id: string | null;
  origin_conversation_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface EventTask {
  id: string;
  event_id: string;
  origin_conversation_id: string | null;
  title: string;
  status:
    | 'open'
    | 'queued'
    | 'in_progress'
    | 'blocked'
    | 'waiting_approval'
    | 'done'
    | 'failed'
    | 'cancelled';
  instructions: string;
  definition_of_done: string;
  priority: 'low' | 'normal' | 'high' | 'urgent';
  due_at: string | null;
  execution_mode: 'manual';
  assigned_agent_id: string | null;
  current_run_id: string | null;
  result_summary: string;
  blocker: string;
  budget_policy: {
    max_steps?: number;
    timeout_seconds?: number;
    max_tool_calls?: number;
    max_cost_usd?: number;
  };
  order: number;
  created_by: string;
  created_at: string;
  updated_at: string;
}

export interface EventProposal {
  id: string;
  event_id: string;
  tool: string;
  args: Record<string, unknown>;
  reason: string;
  impact: string;
  risk: string;
  routing: string;
  status: string;
  expires_at: string | null;
  decided_at: string | null;
  provider_operation_id: string | null;
  receipt: Record<string, unknown> | null;
  error: string | null;
  created_at: string;
  updated_at: string;
}

export interface EventTrailEntry {
  id: string;
  kind: string;
  summary: string;
  actor_id: string | null;
  run_id: string | null;
  proposal_id: string | null;
  task_id: string | null;
  evidence_refs: unknown[];
  payload: Record<string, unknown>;
  created_at: string;
}

export interface EventFile {
  id: string;
  name: string;
  kind: string;
  content_type: string;
  size_bytes: number;
  origin_session_id: string | null;
  origin_run_id: string | null;
  created_at: string;
}

export interface EventSurfaceResponse {
  event: EventSummary;
  surface: {
    type: string;
    sections: Array<
      | { kind: 'status'; summary: string; phase: string }
      | { kind: 'tasks'; tasks: EventTask[] }
      | { kind: 'activity'; entries: EventTrailEntry[] }
      | { kind: 'notes'; notes: string }
      | { kind: 'files'; files: EventFile[] }
    >;
    proposals: EventProposal[];
  };
}

export interface TodayEventGroup {
  event: EventSummary;
  needs_decision: EventProposal[];
  changed_proposals: EventProposal[];
  activity: EventTrailEntry[];
  upcoming: EventTask[];
}

export interface TodayResponse {
  generated_at: string;
  events: TodayEventGroup[];
}

export function listEvents() {
  return apiFetch<EventSummary[]>('/events');
}

export function createEvent(data: {
  title: string;
  summary?: string;
  phase?: string;
  notes?: string;
  origin_conversation_id?: string | null;
}) {
  return apiFetch<EventSummary>('/events', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export function getEventSurface(eventId: string) {
  return apiFetch<EventSurfaceResponse>(`/events/${eventId}`);
}

export function getToday() {
  return apiFetch<TodayResponse>('/today');
}

export function createEventTask(eventId: string, title: string) {
  return apiFetch<EventTask>(`/events/${eventId}/tasks`, {
    method: 'POST',
    body: JSON.stringify({ title }),
  });
}

export function updateEventTask(
  eventId: string,
  taskId: string,
  data: Partial<Pick<EventTask, 'title' | 'status' | 'order'>>,
) {
  return apiFetch<EventTask>(`/events/${eventId}/tasks/${taskId}`, {
    method: 'PATCH',
    body: JSON.stringify(data),
  });
}

export function deleteEventTask(eventId: string, taskId: string) {
  return apiFetch<void>(`/events/${eventId}/tasks/${taskId}`, {
    method: 'DELETE',
  });
}

export function decideEventProposal(
  eventId: string,
  proposalId: string,
  decision: 'approve' | 'reject',
) {
  return apiFetch<{ proposal_id: string; status: string; decision: string }>(
    `/events/${eventId}/proposals/${proposalId}/decision`,
    { method: 'POST', body: JSON.stringify({ decision }) },
  );
}
