import { useSyncExternalStore } from 'react';

const PROTOCOL = '1' as const;
const REQUEST_TIMEOUT_MS = 20_000;
const MAX_ATTEMPTS = 2;

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

export interface SurfaceInteraction {
  id: string;
  event_id: string;
  build_id: string;
  code_revision_id: string;
  name: string;
  interaction_class: string;
  component_id: string;
  status: string;
  handling_run_id: string | null;
  proposal_id: string | null;
  request_message_id: string | null;
  outcome_message_id: string | null;
  result: Record<string, unknown>;
  error: string | null;
  created_at: string;
  updated_at: string;
}

export interface SurfaceContext {
  protocol_version: '1';
  command_contract_version: '1';
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
    interactions: SurfaceInteraction[];
    surface?: Record<string, Array<SurfaceDataRecord>>;
  };
}

export interface SurfaceAction {
  name: string;
  payload: Record<string, unknown>;
  reason?: string;
}

export interface SurfaceCommandOptions {
  componentId?: string;
  idempotencyKey?: string;
}

export type SurfaceRuntimeStatus = 'disconnected' | 'healthy' | 'degraded';

export interface SurfaceRuntimeState {
  status: SurfaceRuntimeStatus;
  message?: string;
}

interface BridgeResponse {
  protocol: '1';
  type: 'response';
  sessionId: string;
  requestId: string;
  ok: boolean;
  result?: unknown;
  error?: string;
  retryable?: boolean;
}

interface PendingRequest {
  method: string;
  params: Record<string, unknown>;
  attempts: number;
  requestId: string;
  timeout: ReturnType<typeof setTimeout> | null;
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
}

export class SurfaceRequestTimeoutError extends Error {
  readonly method: string;
  readonly idempotencyKey?: string;

  constructor(method: string, idempotencyKey?: string) {
    super(`Aloy Surface ${method} request timed out`);
    this.name = 'SurfaceRequestTimeoutError';
    this.method = method;
    this.idempotencyKey = idempotencyKey;
  }
}

let port: MessagePort | null = null;
let sessionId: string | null = null;
let context: SurfaceContext | null = null;
const disconnectedRuntime: SurfaceRuntimeState = { status: 'disconnected' };
let runtime: SurfaceRuntimeState = disconnectedRuntime;
const contextSubscribers = new Set<() => void>();
const runtimeSubscribers = new Set<() => void>();
const pending = new Map<string, PendingRequest>();

function publishContext(next: SurfaceContext) {
  context = next;
  for (const subscriber of contextSubscribers) subscriber();
}

function publishRuntime(next: SurfaceRuntimeState) {
  runtime = next;
  for (const subscriber of runtimeSubscribers) subscriber();
}

function clearPendingTimeout(request: PendingRequest) {
  if (request.timeout) clearTimeout(request.timeout);
  request.timeout = null;
}

function rejectPending(request: PendingRequest, error: Error) {
  clearPendingTimeout(request);
  pending.delete(request.requestId);
  request.reject(error);
}

function send(request: PendingRequest) {
  if (!port || !sessionId) {
    rejectPending(request, new Error('Aloy Surface bridge is not ready'));
    return;
  }
  clearPendingTimeout(request);
  pending.delete(request.requestId);
  request.requestId = crypto.randomUUID();
  request.attempts += 1;
  pending.set(request.requestId, request);
  request.timeout = setTimeout(() => {
    if (!pending.has(request.requestId)) return;
    if (request.attempts < MAX_ATTEMPTS && port && sessionId) {
      send(request);
      return;
    }
    rejectPending(
      request,
      new SurfaceRequestTimeoutError(
        request.method,
        typeof request.params.idempotencyKey === 'string'
          ? request.params.idempotencyKey
          : undefined,
      ),
    );
  }, REQUEST_TIMEOUT_MS);
  try {
    port.postMessage({
      protocol: PROTOCOL,
      type: 'request',
      sessionId,
      requestId: request.requestId,
      method: request.method,
      params: request.params,
    });
  } catch {
    port.close();
    port = null;
    sessionId = null;
    publishRuntime({
      status: 'degraded',
      message: 'The Aloy Surface bridge disconnected',
    });
    // Keep the bounded request pending so an imminent host reconnect can
    // replay it with the same idempotency key.
  }
}

function replayPending() {
  for (const request of [...pending.values()]) {
    if (request.attempts < MAX_ATTEMPTS) send(request);
    else rejectPending(request, new Error('Aloy Surface bridge reconnected after request retry'));
  }
}

function onPortMessage(event: MessageEvent<unknown>) {
  const value = event.data as {
    protocol?: string;
    type?: string;
    sessionId?: string;
    nonce?: string;
    context?: SurfaceContext;
    status?: string;
    message?: string;
  } | null;
  if (
    !value
    || value.protocol !== PROTOCOL
    || value.sessionId !== sessionId
    || !value.type
  ) return;
  if (value.type === 'ping' && typeof value.nonce === 'string') {
    port?.postMessage({
      protocol: PROTOCOL,
      type: 'pong',
      sessionId,
      nonce: value.nonce,
    });
    return;
  }
  if (value.type === 'context' && value.context) {
    publishContext(value.context);
    return;
  }
  if (value.type === 'runtime' && value.status === 'degraded') {
    publishRuntime({ status: 'degraded', message: value.message });
    port?.close();
    port = null;
    sessionId = null;
    return;
  }
  if (value.type !== 'response' || !('requestId' in value)) return;
  const response = value as BridgeResponse;
  if (typeof response.requestId !== 'string') return;
  const request = pending.get(response.requestId);
  if (!request) return;
  clearPendingTimeout(request);
  pending.delete(response.requestId);
  if (response.ok) {
    request.resolve(response.result);
    return;
  }
  if (response.retryable && request.attempts < MAX_ATTEMPTS && port && sessionId) {
    send(request);
    return;
  }
  request.reject(new Error(response.error || 'Aloy Surface request failed'));
}

window.addEventListener('message', (event: MessageEvent<unknown>) => {
  const value = event.data as {
    protocol?: string;
    type?: string;
    sessionId?: string;
    context?: SurfaceContext;
  } | null;
  if (
    !event.isTrusted
    || event.source !== window.parent
    || value?.protocol !== PROTOCOL
    || value.type !== 'aloy.surface.connect'
    || typeof value.sessionId !== 'string'
    || !value.sessionId
    || !value.context
    || event.ports.length !== 1
  ) return;

  port?.close();
  for (const request of pending.values()) clearPendingTimeout(request);
  sessionId = value.sessionId;
  port = event.ports[0];
  port.onmessage = onPortMessage;
  port.start();
  publishContext(value.context);
  publishRuntime({ status: 'healthy' });
  port.postMessage({ protocol: PROTOCOL, type: 'ready', sessionId });
  replayPending();
});

function request<T>(method: string, params: Record<string, unknown>): Promise<T> {
  if (!port || !sessionId) {
    return Promise.reject(new Error('Aloy Surface bridge is not ready'));
  }
  return new Promise<T>((resolve, reject) => {
    const pendingRequest: PendingRequest = {
      method,
      params,
      attempts: 0,
      requestId: '',
      timeout: null,
      resolve: (value) => resolve(value as T),
      reject,
    };
    send(pendingRequest);
  });
}

function componentId(value?: string) {
  return value?.trim().slice(0, 200) || 'surface';
}

function idempotencyKey(value?: string) {
  return value?.trim().slice(0, 200) || crypto.randomUUID();
}

export function subscribeSurface(listener: () => void) {
  contextSubscribers.add(listener);
  return () => contextSubscribers.delete(listener);
}

export function subscribeSurfaceRuntime(listener: () => void) {
  runtimeSubscribers.add(listener);
  return () => runtimeSubscribers.delete(listener);
}

export function getSurfaceContext(): SurfaceContext | null {
  return context;
}

export function getSurfaceRuntime(): SurfaceRuntimeState {
  return runtime;
}

export function useSurfaceContext(): SurfaceContext | null {
  return useSyncExternalStore(subscribeSurface, getSurfaceContext, () => null);
}

export function useSurfaceRuntime(): SurfaceRuntimeState {
  return useSyncExternalStore(
    subscribeSurfaceRuntime,
    getSurfaceRuntime,
    () => disconnectedRuntime,
  );
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

export function useInteractions(): SurfaceInteraction[] {
  return useSurfaceContext()?.data.interactions ?? [];
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

/** Send a manifest-declared command to Aloy's host-owned command runtime. */
export function command<T = Record<string, unknown>>(
  name: string,
  input: Record<string, unknown>,
  options: SurfaceCommandOptions = {},
): Promise<T> {
  return request<T>('command', {
    name,
    payload: input,
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
