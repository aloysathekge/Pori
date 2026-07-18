import {
  useCallback,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from 'react';

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

export interface SurfaceCommandAttempt {
  id: string;
  event_id: string;
  build_id: string;
  code_revision_id: string;
  interaction_id: string | null;
  method: string;
  name: string;
  interaction_class: string;
  component_id: string;
  base_data_revision: number;
  observed_data_revision: number;
  status: string;
  error_code: string | null;
  error: string | null;
  http_status: number;
  retryable: boolean;
  created_at: string;
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
    command_attempts?: SurfaceCommandAttempt[];
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

export type SurfaceRequestErrorCode =
  | 'conflict'
  | 'permission_denied'
  | 'invalid'
  | 'rate_limited'
  | 'unavailable'
  | 'timeout'
  | 'disconnected'
  | 'failed';

export type SurfaceCommandStatus =
  | 'idle'
  | 'pending'
  | 'committed'
  | 'accepted'
  | 'conflict'
  | 'failed';

export interface SurfaceCommandReceipt {
  id: string;
  status: string;
  name: string;
  interaction_class: string;
  data_revision: number | null;
  handling_run_id: string | null;
  proposal_id: string | null;
  result: Record<string, unknown>;
  replayed: boolean;
}

export interface SurfaceCommandFeedbackProps {
  role: 'status' | 'alert';
  'aria-live': 'polite' | 'assertive';
  'aria-atomic': true;
  'data-aloy-command-name': string;
  'data-aloy-command-status': SurfaceCommandStatus;
}

export interface SurfaceCommandController<
  TInput extends Record<string, unknown>,
  TResult,
> {
  status: SurfaceCommandStatus;
  pending: boolean;
  result: TResult | null;
  error: SurfaceRequestError | null;
  execute: (input: TInput) => Promise<TResult>;
  retry: () => Promise<TResult>;
  reset: () => void;
  feedbackProps: SurfaceCommandFeedbackProps;
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
  errorCode?: SurfaceRequestErrorCode;
  serverCode?: string;
  attemptId?: string;
  statusCode?: number;
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

export class SurfaceRequestError extends Error {
  readonly method: string;
  readonly code: SurfaceRequestErrorCode;
  readonly retryable: boolean;
  readonly statusCode?: number;
  readonly idempotencyKey?: string;
  readonly serverCode?: string;
  readonly attemptId?: string;

  constructor(
    message: string,
    options: {
      method: string;
      code?: SurfaceRequestErrorCode;
      retryable?: boolean;
      statusCode?: number;
      idempotencyKey?: string;
      serverCode?: string;
      attemptId?: string;
    },
  ) {
    super(message);
    this.name = 'SurfaceRequestError';
    this.method = options.method;
    this.code = options.code ?? 'failed';
    this.retryable = options.retryable ?? false;
    this.statusCode = options.statusCode;
    this.idempotencyKey = options.idempotencyKey;
    this.serverCode = options.serverCode;
    this.attemptId = options.attemptId;
  }
}

export class SurfaceRequestTimeoutError extends SurfaceRequestError {
  readonly method: string;
  readonly idempotencyKey?: string;

  constructor(method: string, idempotencyKey?: string) {
    super(`Aloy Surface ${method} request timed out`, {
      method,
      code: 'timeout',
      retryable: true,
      idempotencyKey,
    });
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
    rejectPending(
      request,
      new SurfaceRequestError('Aloy Surface bridge is not ready', {
        method: request.method,
        code: 'disconnected',
        retryable: true,
        idempotencyKey:
          typeof request.params.idempotencyKey === 'string'
            ? request.params.idempotencyKey
            : undefined,
      }),
    );
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
    else {
      rejectPending(
        request,
        new SurfaceRequestError(
          'Aloy Surface bridge reconnected after request retry',
          {
            method: request.method,
            code: 'unavailable',
            retryable: true,
            idempotencyKey:
              typeof request.params.idempotencyKey === 'string'
                ? request.params.idempotencyKey
                : undefined,
          },
        ),
      );
    }
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
  request.reject(
    new SurfaceRequestError(
      response.error || 'Aloy Surface request failed',
      {
        method: request.method,
        code: response.errorCode,
        retryable: response.retryable,
        statusCode: response.statusCode,
        serverCode: response.serverCode,
        attemptId: response.attemptId,
        idempotencyKey:
          typeof request.params.idempotencyKey === 'string'
            ? request.params.idempotencyKey
            : undefined,
      },
    ),
  );
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
    return Promise.reject(
      new SurfaceRequestError('Aloy Surface bridge is not ready', {
        method,
        code: 'disconnected',
        retryable: true,
        idempotencyKey:
          typeof params.idempotencyKey === 'string'
            ? params.idempotencyKey
            : undefined,
      }),
    );
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

export function useCommandAttempts(): SurfaceCommandAttempt[] {
  return useSurfaceContext()?.data.command_attempts ?? [];
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

function commandStatus(result: unknown): SurfaceCommandStatus {
  if (
    result
    && typeof result === 'object'
    && !Array.isArray(result)
    && (result as { status?: unknown }).status === 'committed'
  ) {
    return 'committed';
  }
  return 'accepted';
}

function requestError(cause: unknown, method: string): SurfaceRequestError {
  if (cause instanceof SurfaceRequestError) return cause;
  return new SurfaceRequestError(
    cause instanceof Error ? cause.message : 'Aloy Surface request failed',
    { method },
  );
}

function snapshotCommandInput<TInput extends Record<string, unknown>>(
  input: TInput,
): TInput {
  try {
    return JSON.parse(JSON.stringify(input)) as TInput;
  } catch {
    throw new SurfaceRequestError(
      'Surface command input must be JSON-serializable',
      { method: 'command', code: 'invalid' },
    );
  }
}

/**
 * Host-owned command lifecycle for generated React controls.
 *
 * A new execute call receives a new idempotency key. Retry reuses the exact
 * input and key, while duplicate submits share the in-flight Promise. The
 * command is reported as committed only after the host has acknowledged its
 * durable state result and delivered refreshed canonical context.
 */
export function useSurfaceCommand<
  TInput extends Record<string, unknown> = Record<string, unknown>,
  TResult = SurfaceCommandReceipt,
>(
  name: string,
  options: Pick<SurfaceCommandOptions, 'componentId'> = {},
): SurfaceCommandController<TInput, TResult> {
  const [status, setStatus] = useState<SurfaceCommandStatus>('idle');
  const [result, setResult] = useState<TResult | null>(null);
  const [error, setError] = useState<SurfaceRequestError | null>(null);
  const lastRequest = useRef<{ input: TInput; idempotencyKey: string } | null>(
    null,
  );
  const inFlight = useRef<Promise<TResult> | null>(null);

  const run = useCallback(
    (input: TInput, requestId: string): Promise<TResult> => {
      if (inFlight.current) return inFlight.current;
      setStatus('pending');
      setError(null);
      const current = command<TResult>(name, input, {
        componentId: options.componentId,
        idempotencyKey: requestId,
      })
        .then((receipt) => {
          setResult(receipt);
          setStatus(commandStatus(receipt));
          return receipt;
        })
        .catch((cause: unknown) => {
          const nextError = requestError(cause, 'command');
          setError(nextError);
          setStatus(nextError.code === 'conflict' ? 'conflict' : 'failed');
          throw nextError;
        })
        .finally(() => {
          if (inFlight.current === current) inFlight.current = null;
        });
      inFlight.current = current;
      return current;
    },
    [name, options.componentId],
  );

  const execute = useCallback(
    (input: TInput) => {
      if (inFlight.current) return inFlight.current;
      let snapshot: TInput;
      try {
        snapshot = snapshotCommandInput(input);
      } catch (cause) {
        const nextError = requestError(cause, 'command');
        setError(nextError);
        setStatus('failed');
        return Promise.reject(nextError);
      }
      const requestId = idempotencyKey();
      lastRequest.current = { input: snapshot, idempotencyKey: requestId };
      return run(snapshot, requestId);
    },
    [run],
  );

  const retry = useCallback(() => {
    if (!lastRequest.current) {
      return Promise.reject(
        new SurfaceRequestError('No Surface command is available to retry', {
          method: 'command',
          code: 'failed',
        }),
      );
    }
    return run(lastRequest.current.input, lastRequest.current.idempotencyKey);
  }, [run]);

  const reset = useCallback(() => {
    if (inFlight.current) return;
    lastRequest.current = null;
    setStatus('idle');
    setResult(null);
    setError(null);
  }, []);

  const feedbackProps = useMemo<SurfaceCommandFeedbackProps>(
    () => ({
      role: status === 'failed' || status === 'conflict' ? 'alert' : 'status',
      'aria-live':
        status === 'failed' || status === 'conflict' ? 'assertive' : 'polite',
      'aria-atomic': true,
      'data-aloy-command-name': name,
      'data-aloy-command-status': status,
    }),
    [name, status],
  );

  return {
    status,
    pending: status === 'pending',
    result,
    error,
    execute,
    retry,
    reset,
    feedbackProps,
  };
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
