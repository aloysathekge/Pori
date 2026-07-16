import { useSyncExternalStore } from 'react';

export interface SurfaceDataRecord<T = Record<string, unknown>> {
  id: string;
  namespace: string;
  key: string;
  data: T;
  revision: number;
  posture:
    | 'official'
    | 'user_reported'
    | 'inferred'
    | 'estimated'
    | 'pending'
    | 'committed'
    | 'failed'
    | 'indeterminate';
  provenance: Record<string, unknown>;
  evidence_refs: Array<Record<string, unknown>>;
}

export interface SurfaceContext {
  protocol_version: '1';
  sdk_version: '1';
  event_id: string;
  project_id: string;
  build_id: string;
  code_revision_id: string;
  data_revision: number;
  capabilities: string[];
  widgets: string[];
  data: {
    event?: Record<string, unknown>;
    tasks?: Array<Record<string, unknown>>;
    files?: Array<Record<string, unknown>>;
    proposals?: Array<Record<string, unknown>>;
    receipts?: Array<Record<string, unknown>>;
    trail?: Array<Record<string, unknown>>;
    surface?: Record<string, Array<SurfaceDataRecord>>;
  };
}

export interface SurfaceAction {
  name: string;
  payload: Record<string, unknown>;
  reason?: string;
}

interface BridgeResponse {
  protocol: '1';
  type: 'response';
  requestId: string;
  ok: boolean;
  result?: unknown;
  error?: string;
}

let port: MessagePort | null = null;
let context: SurfaceContext | null = null;
const subscribers = new Set<() => void>();
const pending = new Map<
  string,
  { resolve: (value: unknown) => void; reject: (error: Error) => void }
>();

function publish(next: SurfaceContext) {
  context = next;
  for (const subscriber of subscribers) subscriber();
}

function onPortMessage(event: MessageEvent<unknown>) {
  const value = event.data as BridgeResponse | { type?: string; context?: SurfaceContext };
  if (value?.type === 'context' && value.context) {
    publish(value.context);
    return;
  }
  if (value?.type !== 'response' || !('requestId' in value)) return;
  const response = value as BridgeResponse;
  if (typeof response.requestId !== 'string') return;
  const waiter = pending.get(response.requestId);
  if (!waiter) return;
  pending.delete(response.requestId);
  if (response.ok) waiter.resolve(response.result);
  else waiter.reject(new Error(response.error || 'Aloy Surface request failed'));
}

window.addEventListener('message', (event: MessageEvent<unknown>) => {
  const value = event.data as {
    protocol?: string;
    type?: string;
    context?: SurfaceContext;
  } | null;
  if (
    !event.isTrusted
    || event.source !== window.parent
    || value?.protocol !== '1'
    || value.type !== 'aloy.surface.connect'
    || !value.context
    || event.ports.length !== 1
  ) return;
  port?.close();
  for (const waiter of pending.values()) {
    waiter.reject(new Error('Aloy Surface bridge reconnected'));
  }
  pending.clear();
  port = event.ports[0];
  port.onmessage = onPortMessage;
  port.start();
  publish(value.context);
});

function request<T>(method: string, params: Record<string, unknown>): Promise<T> {
  if (!port) return Promise.reject(new Error('Aloy Surface bridge is not ready'));
  const requestId = crypto.randomUUID();
  return new Promise<T>((resolve, reject) => {
    pending.set(requestId, {
      resolve: (value) => resolve(value as T),
      reject,
    });
    port?.postMessage({ protocol: '1', type: 'request', requestId, method, params });
  });
}

function componentId(value?: string) {
  return value?.trim().slice(0, 200) || 'surface';
}

function idempotencyKey(value?: string) {
  return value?.trim().slice(0, 200) || crypto.randomUUID();
}

export function subscribeSurface(listener: () => void) {
  subscribers.add(listener);
  return () => subscribers.delete(listener);
}

export function getSurfaceContext(): SurfaceContext | null {
  return context;
}

export function useSurfaceContext(): SurfaceContext | null {
  return useSyncExternalStore(subscribeSurface, getSurfaceContext, () => null);
}

export function useEvent<T = Record<string, unknown>>(): T | null {
  return (useSurfaceContext()?.data.event as T | undefined) ?? null;
}

export function useSurfaceData<T = Record<string, unknown>>(
  namespace: string,
): Array<SurfaceDataRecord<T>> {
  const records = useSurfaceContext()?.data.surface?.[namespace] ?? [];
  return records as Array<SurfaceDataRecord<T>>;
}

export function useTasks(): Array<Record<string, unknown>> {
  return useSurfaceContext()?.data.tasks ?? [];
}

export function dispatch<T = Record<string, unknown>>(
  name: string,
  payload: Record<string, unknown>,
  options: { componentId?: string; idempotencyKey?: string } = {},
): Promise<T> {
  return request<T>('dispatch', {
    name,
    payload,
    componentId: componentId(options.componentId),
    idempotencyKey: idempotencyKey(options.idempotencyKey),
  });
}

export function askAloy<T = Record<string, unknown>>(
  message: string,
  surfaceContext: Record<string, unknown> = {},
  options: { componentId?: string; idempotencyKey?: string } = {},
): Promise<T> {
  return request<T>('askAloy', {
    message,
    context: surfaceContext,
    componentId: componentId(options.componentId),
    idempotencyKey: idempotencyKey(options.idempotencyKey),
  });
}

export function requestAction<T = Record<string, unknown>>(
  action: SurfaceAction,
  options: { componentId?: string; idempotencyKey?: string } = {},
): Promise<T> {
  return request<T>('requestAction', {
    action,
    componentId: componentId(options.componentId),
    idempotencyKey: idempotencyKey(options.idempotencyKey),
  });
}
