import { apiFetch, apiStreamFetch } from './client';

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

export interface TaskExecutionResponse {
  task: EventTask;
  run: {
    id: string;
    status: string;
    conversation_id: string | null;
    attempt_count: number;
  } | null;
  idempotent: boolean;
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

export interface EventExecutionGroup {
  id: string;
  task_id: string;
  task_title: string;
  task_status: EventTask['status'];
  run_id: string;
  run_status: string;
  conversation_id: string | null;
  created_at: string;
  completed_at: string | null;
  entries: EventTrailEntry[];
  artifacts: EventFile[];
  proposals: EventProposal[];
  receipts: Array<Record<string, unknown>>;
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
      | { kind: 'activity'; entries: EventTrailEntry[]; next_cursor: string | null }
      | { kind: 'notes'; notes: string }
      | { kind: 'files'; files: EventFile[] }
    >;
    proposals: EventProposal[];
    execution_groups: EventExecutionGroup[];
  };
}

export interface TodayEventGroup {
  event: EventSummary;
  needs_decision: EventProposal[];
  changed_proposals: EventProposal[];
  activity: EventTrailEntry[];
  upcoming: EventTask[];
  blocked: EventTask[];
  stale: EventTask[];
}

export interface TodayNotification {
  id: string;
  kind: string;
  title: string;
  summary: string;
  event_id: string;
  event_title: string;
  event_is_life: boolean;
  proposal_id: string | null;
  task_id: string | null;
  run_id: string | null;
  status: string | null;
  created_at: string;
}

export interface TodayResponse {
  generated_at: string;
  notifications: TodayNotification[];
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

export function getEventTrail(eventId: string, cursor: string, limit = 50) {
  const params = new URLSearchParams({ cursor, limit: String(limit) });
  return apiFetch<{ entries: EventTrailEntry[]; next_cursor: string | null }>(
    `/events/${eventId}/trail?${params.toString()}`,
  );
}

export interface EventLiveChange {
  event_id: string;
  conversation_id: string | null;
  entry: EventTrailEntry;
}

export async function streamEventChanges(
  eventId: string,
  cursor: string | null,
  callbacks: {
    onCursor: (cursor: string | null) => void;
    onChange: (change: EventLiveChange) => void;
    onHeartbeat: () => void;
  },
  signal: AbortSignal,
) {
  const query = cursor ? `?cursor=${encodeURIComponent(cursor)}` : '';
  const response = await apiStreamFetch(
    `/events/${eventId}/live${query}`,
    undefined,
    signal,
    'GET',
  );
  if (!response.body) throw new Error('Event stream has no response body');
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
    let boundary = buffer.indexOf('\n\n');
    while (boundary !== -1) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      let event = 'message';
      let id: string | null = null;
      const data: string[] = [];
      for (const line of frame.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim();
        if (line.startsWith('id:')) id = line.slice(3).trim();
        if (line.startsWith('data:')) data.push(line.slice(5).trimStart());
      }
      if (data.length) {
        const payload = JSON.parse(data.join('\n')) as Record<string, unknown>;
        if (event === 'ready') callbacks.onCursor((payload.cursor as string | null) ?? null);
        if (event === 'event_change') {
          if (id) callbacks.onCursor(id);
          callbacks.onChange(payload as unknown as EventLiveChange);
        }
        if (event === 'heartbeat') callbacks.onHeartbeat();
      }
      boundary = buffer.indexOf('\n\n');
    }
  }
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

export function workOnEventTask(eventId: string, taskId: string) {
  return apiFetch<TaskExecutionResponse>(`/events/${eventId}/tasks/${taskId}/work`, {
    method: 'POST',
  });
}

export function stopEventTask(eventId: string, taskId: string) {
  return apiFetch<TaskExecutionResponse>(`/events/${eventId}/tasks/${taskId}/stop`, {
    method: 'POST',
  });
}

export function retryEventTask(eventId: string, taskId: string) {
  return apiFetch<TaskExecutionResponse>(`/events/${eventId}/tasks/${taskId}/retry`, {
    method: 'POST',
  });
}

export function resumeEventTask(eventId: string, taskId: string, response?: string) {
  return apiFetch<TaskExecutionResponse>(`/events/${eventId}/tasks/${taskId}/resume`, {
    method: 'POST',
    body: JSON.stringify({ response: response || null }),
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
