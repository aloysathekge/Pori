import { ApiError, apiFetch, apiStreamFetch } from './client';
import {
  CLARIFICATION_REQUEST,
  TEXT_DELTA,
  THINKING_DELTA,
  TOOL_CALL_END,
  TOOL_CALL_START,
} from '@pori/client';
import type { ClarificationRequestPayload } from '@pori/client';
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
  onStep?: (info: { step: number; max_steps: number }) => void;
  onClarification?: (request: ClarificationRequestPayload) => void;
  onMessage?: (data: SSEMessageEvent) => void;
  onError?: (err: string) => void;
  onDone?: () => void;
}

/** Abort if no bytes arrive for this long (server keepalives every ~15s, so a
 *  healthy stream never goes quiet this long — this catches silent stalls). */
const IDLE_TIMEOUT_MS = 90_000;

export async function streamMessage(
  conversationId: string,
  content: string,
  callbacks: SSECallbacks,
  options?: {
    max_steps?: number;
    team_id?: string | null;
    images?: { data: string; media_type: string }[];
    files?: { name: string; content: string }[];
    documents?: { name: string; data: string; media_type: string }[];
    signal?: AbortSignal;
  },
) {
  const { signal, ...body } = options ?? {};
  await consumeStream(
    (watchdogSignal) =>
      apiStreamFetch(
        `/conversations/${conversationId}/messages`,
        { content, stream: true, ...body },
        watchdogSignal,
      ),
    callbacks,
    signal,
  );
}

/**
 * Re-attach to a conversation's in-flight run (after navigating away and
 * back): the server replays every frame so far, then continues live.
 * Returns false when there is no live run.
 */
export async function attachLiveRun(
  conversationId: string,
  callbacks: SSECallbacks,
  signal?: AbortSignal,
  onAttached?: () => void,
): Promise<boolean> {
  try {
    await consumeStream(
      async (watchdogSignal) => {
        const res = await apiStreamFetch(
          `/conversations/${conversationId}/live`,
          undefined,
          watchdogSignal,
          'GET',
        );
        onAttached?.();
        return res;
      },
      callbacks,
      signal,
    );
    return true;
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) return false; // no live run
    throw err;
  }
}

/** Shared SSE pump: watchdog, CRLF tolerance, trailing flush, single onDone. */
async function consumeStream(
  open: (watchdogSignal: AbortSignal) => Promise<Response>,
  callbacks: SSECallbacks,
  signal?: AbortSignal,
): Promise<void> {
  // Idle watchdog: chain the caller's signal with our own timeout-based abort
  // so a stalled-open connection can't leave the UI in 'sending' forever.
  const watchdog = new AbortController();
  const onCallerAbort = () => watchdog.abort();
  signal?.addEventListener('abort', onCallerAbort);
  let idleTimer: ReturnType<typeof setTimeout> | undefined;
  const resetIdle = () => {
    clearTimeout(idleTimer);
    idleTimer = setTimeout(() => watchdog.abort(), IDLE_TIMEOUT_MS);
  };

  // onDone must fire exactly once (the server also sends an explicit 'done'
  // frame — without this guard both the frame and stream-end would fire it).
  let doneFired = false;
  const wrapped: SSECallbacks = {
    ...callbacks,
    onDone: () => {
      if (doneFired) return;
      doneFired = true;
      callbacks.onDone?.();
    },
  };

  try {
    const res = await open(watchdog.signal);

    const reader = res.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    resetIdle();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      resetIdle();
      // Normalize CRLF so proxy-injected \r\n framing still parses.
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let boundary: number;
      while ((boundary = buffer.indexOf('\n\n')) !== -1) {
        dispatchFrame(buffer.slice(0, boundary), wrapped);
        buffer = buffer.slice(boundary + 2);
      }
    }

    // Flush a trailing frame that arrived without its terminating blank line —
    // otherwise the final 'message' frame (the assistant reply) is dropped.
    if (buffer.trim()) {
      dispatchFrame(buffer, wrapped);
    }
  } catch (err) {
    // A deliberate abort (conversation switch / unmount / watchdog) is not an
    // application error; surface everything else.
    if (!watchdog.signal.aborted) throw err;
    if (!signal?.aborted) {
      // Watchdog fired on its own: the stream stalled silently.
      wrapped.onError?.('The response stream stalled. Please try again.');
    }
  } finally {
    clearTimeout(idleTimer);
    signal?.removeEventListener('abort', onCallerAbort);
  }

  wrapped.onDone?.();
}

/** Resolve a paused `ask_user` (the clarify button bridge). */
export async function submitClarification(
  clarificationId: string,
  value: string,
): Promise<void> {
  await apiFetch(`/conversations/clarify/${encodeURIComponent(clarificationId)}`, {
    method: 'POST',
    body: JSON.stringify({ value }),
  });
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
    case 'step_start':
      cb.onStep?.({
        step: Number(payload.step ?? 0),
        max_steps: Number(payload.max_steps ?? 0),
      });
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
