import { apiStreamFetch } from './client';
import type {
  SSEStatusEvent,
  SSEStepEvent,
  SSEToolEvent,
  SSEMessageEvent,
} from '@/types';

export interface SSECallbacks {
  onStatus?: (data: SSEStatusEvent) => void;
  onTool?: (data: SSEToolEvent) => void;
  onStep?: (data: SSEStepEvent) => void;
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
  const res = await apiStreamFetch(
    `/conversations/${conversationId}/messages`,
    { content, stream: true, ...options },
  );

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    let currentEvent = '';
    for (const line of lines) {
      if (line.startsWith('event: ')) {
        currentEvent = line.slice(7).trim();
      } else if (line.startsWith('data: ')) {
        const jsonStr = line.slice(6);
        try {
          const data = JSON.parse(jsonStr);
          switch (currentEvent) {
            case 'status':
              callbacks.onStatus?.(data as SSEStatusEvent);
              break;
            case 'tool':
              callbacks.onTool?.(data as SSEToolEvent);
              break;
            case 'step':
              callbacks.onStep?.(data as SSEStepEvent);
              break;
            case 'message':
              callbacks.onMessage?.(data as SSEMessageEvent);
              break;
            case 'error':
              callbacks.onError?.(data.detail || 'Unknown streaming error');
              break;
            case 'done':
              callbacks.onDone?.();
              break;
          }
        } catch {
          // skip malformed JSON
        }
      }
    }
  }

  callbacks.onDone?.();
}
