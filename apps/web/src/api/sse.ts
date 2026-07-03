import { apiStreamFetch } from './client';
import {
  CLARIFICATION_REQUEST,
  TEXT_DELTA,
  THINKING_DELTA,
  TOOL_CALL_END,
  TOOL_CALL_START,
} from '@aloy/shared';
import type { ClarificationRequestPayload } from '@aloy/shared';
import type { SSEMessageEvent } from '@/types';

/**
 * Consumes the backend's kernel `PoriEvent` SSE stream (see
 * `products/aloy/backend/pori_cloud/streaming.py`). PoriEvent frames carry
 * `data: {payload, step}`; the final `message` frame is flat. Auth stays the
 * app's Supabase Bearer (via `apiStreamFetch`).
 */
export interface SSECallbacks {
  onText?: (text: string) => void;
  onThinking?: (text: string) => void;
  onToolStart?: (payload: { name?: string; [k: string]: unknown }) => void;
  onToolEnd?: (payload: {
    name?: string;
    success?: boolean;
    [k: string]: unknown;
  }) => void;
  onClarification?: (request: ClarificationRequestPayload) => void;
  onMessage?: (data: SSEMessageEvent) => void;
  onError?: (err: string) => void;
  onDone?: () => void;
}

export async function streamMessage(
  conversationId: string,
  content: string,
  callbacks: SSECallbacks,
  options?: { max_steps?: number; team_id?: string | null },
) {
  const res = await apiStreamFetch(`/conversations/${conversationId}/messages`, {
    content,
    stream: true,
    ...options,
  });

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary: number;
    while ((boundary = buffer.indexOf('\n\n')) !== -1) {
      dispatchFrame(buffer.slice(0, boundary), callbacks);
      buffer = buffer.slice(boundary + 2);
    }
  }

  callbacks.onDone?.();
}

function dispatchFrame(frame: string, cb: SSECallbacks) {
  let event = 'message';
  const dataLines: string[] = [];
  for (const line of frame.split('\n')) {
    if (line.startsWith(':')) continue; // keepalive comment
    if (line.startsWith('event:')) event = line.slice(6).trim();
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).replace(/^ /, ''));
  }
  if (dataLines.length === 0) return;

  let data: Record<string, unknown>;
  try {
    data = JSON.parse(dataLines.join('\n'));
  } catch {
    return; // skip malformed JSON
  }
  const payload = (data.payload as Record<string, unknown>) ?? {};

  switch (event) {
    case TEXT_DELTA:
      cb.onText?.(String(payload.text ?? ''));
      break;
    case THINKING_DELTA:
      cb.onThinking?.(String(payload.text ?? ''));
      break;
    case TOOL_CALL_START:
      cb.onToolStart?.(payload);
      break;
    case TOOL_CALL_END:
      cb.onToolEnd?.(payload);
      break;
    case CLARIFICATION_REQUEST:
      cb.onClarification?.(payload as unknown as ClarificationRequestPayload);
      break;
    case 'message':
      cb.onMessage?.(data as unknown as SSEMessageEvent);
      break;
    case 'error':
      cb.onError?.(String(data.detail ?? 'Unknown streaming error'));
      break;
    case 'done':
      cb.onDone?.();
      break;
    // run_start / run_end / step_* / status / llm_retry: no-op for now
  }
}
