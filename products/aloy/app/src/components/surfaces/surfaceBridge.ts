import {
  createSurfaceInteraction,
  getSurfaceRuntimeContext,
  type SurfaceInteractionMethod,
  type SurfaceInteractionRequest,
  type SurfaceInteractionResponse,
  type SurfaceResourceRef,
  type SurfaceRuntimeContext,
} from '../../api/surfaces';
import { ApiError } from '../../api/client';

const PROTOCOL = '1' as const;
const MAX_IN_FLIGHT = 8;

export type SurfaceBridgeStatus =
  | 'connecting'
  | 'healthy'
  | 'degraded'
  | 'disconnected';

type SurfaceBridgeErrorCode =
  | 'conflict'
  | 'permission_denied'
  | 'invalid'
  | 'rate_limited'
  | 'unavailable'
  | 'timeout'
  | 'disconnected'
  | 'failed';

export interface SurfaceBridgeStatusUpdate {
  status: SurfaceBridgeStatus;
  message?: string;
}

export interface SurfaceAloyHandoff {
  method: SurfaceInteractionMethod;
  name: string;
  componentId: string;
  message?: string;
  reason?: string;
  resourceRefs: SurfaceResourceRef[];
  response: SurfaceInteractionResponse;
}

export interface SurfaceResourceOpenRequest {
  fileId: string;
  componentId: string;
}

export interface SurfaceElementSelection {
  selectionId: string;
  buildId: string;
  codeRevisionId: string;
  nodeId: string;
  tagName: string;
  role: string;
  accessibleName: string;
  text: string;
  componentId: string;
  resource: string | null;
  source: string | null;
  bounds: { x: number; y: number; width: number; height: number };
  styles: {
    display: string;
    color: string;
    backgroundColor: string;
    fontSize: string;
  };
}

export function shouldSummonAloy(
  method: SurfaceInteractionMethod,
  response: SurfaceInteractionResponse,
): boolean {
  return (
    method === 'ask_aloy'
    || method === 'request_action'
    || Boolean(response.handling_run_id)
    || Boolean(response.proposal_id)
  );
}

export interface SurfaceBridgeOptions {
  onStatus?: (update: SurfaceBridgeStatusUpdate) => void;
  onAloyHandoff?: (handoff: SurfaceAloyHandoff) => void;
  onOpenResource?: (request: SurfaceResourceOpenRequest) => void | Promise<void>;
  onElementSelection?: (selection: SurfaceElementSelection) => void;
  contextTimeoutMs?: number;
  handshakeTimeoutMs?: number;
  heartbeatIntervalMs?: number;
  heartbeatGraceMs?: number;
  requestTimeoutMs?: number;
}

export interface SurfaceBridgeDependencies {
  getContext: (
    eventId: string,
    buildId: string,
    signal?: AbortSignal,
  ) => Promise<SurfaceRuntimeContext>;
  createInteraction: (
    eventId: string,
    request: SurfaceInteractionRequest,
    signal?: AbortSignal,
  ) => Promise<SurfaceInteractionResponse>;
  createChannel: () => MessageChannel;
  now: () => number;
  randomId: () => string;
}

type BridgeRequest = {
  protocol: '1';
  type: 'request';
  sessionId: string;
  requestId: string;
  method:
    | 'getContext'
    | 'command'
    | 'dispatch'
    | 'askAloy'
    | 'requestAction'
    | 'openResource';
  params?: unknown;
};

type InFlight = {
  controller: AbortController;
  timeout: ReturnType<typeof setTimeout>;
};

const DEFAULT_DEPENDENCIES: SurfaceBridgeDependencies = {
  getContext: getSurfaceRuntimeContext,
  createInteraction: createSurfaceInteraction,
  createChannel: () => new MessageChannel(),
  now: () => Date.now(),
  randomId: () => crypto.randomUUID(),
};

function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('Surface bridge parameters must be an object');
  }
  return value as Record<string, unknown>;
}

function string(value: unknown, name: string, max = 50_000): string {
  if (typeof value !== 'string' || !value.trim() || value.length > max) {
    throw new Error(`Surface bridge ${name} is invalid`);
  }
  return value.trim();
}

function payload(value: unknown): Record<string, unknown> {
  const result = object(value ?? {});
  if (JSON.stringify(result).length > 128 * 1024) {
    throw new Error('Surface bridge payload is too large');
  }
  return result;
}

function resourceRefs(value: unknown): SurfaceResourceRef[] {
  if (value === undefined) return [];
  if (!Array.isArray(value) || value.length > 20) {
    throw new Error('Surface bridge resource references are invalid');
  }
  const refs = value.map((candidate) => {
    const ref = object(candidate);
    if (ref.type !== 'file') {
      throw new Error('Surface bridge resource type is invalid');
    }
    return { type: 'file' as const, id: string(ref.id, 'resource id', 200) };
  });
  if (new Set(refs.map((ref) => `${ref.type}:${ref.id}`)).size !== refs.length) {
    throw new Error('Surface bridge resource references must be unique');
  }
  return refs;
}

function errorMessage(cause: unknown, fallback: string): string {
  if (cause instanceof Error && cause.name === 'AbortError') return fallback;
  return cause instanceof Error ? cause.message : fallback;
}

function apiErrorDetails(cause: unknown): Record<string, unknown> {
  return cause instanceof ApiError
    && cause.details
    && typeof cause.details === 'object'
    && !Array.isArray(cause.details)
    ? cause.details as Record<string, unknown>
    : {};
}

function retryableFailure(cause: unknown, controller: AbortController): boolean {
  const declared = apiErrorDetails(cause).retryable;
  if (typeof declared === 'boolean') return declared;
  return (
    controller.signal.aborted
    || cause instanceof TypeError
    || (cause instanceof ApiError && [409, 429, 503].includes(cause.status))
  );
}

function failureCode(
  cause: unknown,
  controller: AbortController,
): SurfaceBridgeErrorCode {
  if (controller.signal.aborted) return 'timeout';
  if (cause instanceof TypeError) return 'unavailable';
  if (!(cause instanceof ApiError)) return 'failed';
  if (cause.status === 409) return 'conflict';
  if (cause.status === 401 || cause.status === 403) return 'permission_denied';
  if (cause.status === 422) return 'invalid';
  if (cause.status === 429) return 'rate_limited';
  if (cause.status === 503) return 'unavailable';
  return 'failed';
}

export class SurfaceBridgeHost {
  private port: MessagePort | null = null;
  private context: SurfaceRuntimeContext | null = null;
  private sessionId: string | null = null;
  private status: SurfaceBridgeStatus = 'disconnected';
  private generation = 0;
  private inFlight = new Map<string, InFlight>();
  private contextController: AbortController | null = null;
  private contextTimeout: ReturnType<typeof setTimeout> | null = null;
  private handshakeTimeout: ReturnType<typeof setTimeout> | null = null;
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null;
  private pendingPing: { nonce: string; sentAt: number } | null = null;
  private inspectionEnabled = false;
  private readyResolve: (() => void) | null = null;
  private readyReject: ((error: Error) => void) | null = null;
  private readonly eventId: string;
  private readonly buildId: string;
  private readonly options: Required<
    Omit<SurfaceBridgeOptions, 'onStatus' | 'onAloyHandoff' | 'onOpenResource' | 'onElementSelection'>
  > & Pick<SurfaceBridgeOptions, 'onStatus' | 'onAloyHandoff' | 'onOpenResource' | 'onElementSelection'>;
  private readonly dependencies: SurfaceBridgeDependencies;

  constructor(
    eventId: string,
    buildId: string,
    options: SurfaceBridgeOptions = {},
    dependencies: Partial<SurfaceBridgeDependencies> = {},
  ) {
    this.eventId = eventId;
    this.buildId = buildId;
    this.options = {
      onStatus: options.onStatus,
      onAloyHandoff: options.onAloyHandoff,
      onOpenResource: options.onOpenResource,
      onElementSelection: options.onElementSelection,
      contextTimeoutMs: options.contextTimeoutMs ?? 12_000,
      handshakeTimeoutMs: options.handshakeTimeoutMs ?? 5_000,
      heartbeatIntervalMs: options.heartbeatIntervalMs ?? 5_000,
      heartbeatGraceMs: options.heartbeatGraceMs ?? 15_000,
      requestTimeoutMs: options.requestTimeoutMs ?? 20_000,
    };
    this.dependencies = { ...DEFAULT_DEPENDENCIES, ...dependencies };
  }

  get currentStatus(): SurfaceBridgeStatus {
    return this.status;
  }

  async connect(frame: HTMLIFrameElement): Promise<void> {
    this.teardown(new Error('Surface bridge reconnected'));
    const generation = ++this.generation;
    this.emit({ status: 'connecting', message: 'Connecting Surface to Aloy' });
    const frameWindow = frame.contentWindow;
    if (!frameWindow) {
      const error = new Error('Surface frame is unavailable');
      this.fail(error.message);
      throw error;
    }

    try {
      this.context = await this.loadContext();
    } catch (cause) {
      if (generation !== this.generation) throw cause;
      const message = errorMessage(cause, 'Surface context request timed out');
      this.fail(message);
      throw new Error(message);
    }
    if (generation !== this.generation) {
      throw new Error('Surface bridge connection was superseded');
    }

    const channel = this.dependencies.createChannel();
    const sessionId = this.dependencies.randomId();
    this.sessionId = sessionId;
    this.port = channel.port1;
    this.port.onmessage = (event: MessageEvent<unknown>) => {
      void this.handle(event.data, generation);
    };
    this.port.start();

    const ready = new Promise<void>((resolve, reject) => {
      this.readyResolve = resolve;
      this.readyReject = reject;
      this.handshakeTimeout = setTimeout(() => {
        reject(new Error('Surface did not acknowledge the secure bridge'));
      }, this.options.handshakeTimeoutMs);
    });

    frameWindow.postMessage(
      {
        protocol: PROTOCOL,
        type: 'aloy.surface.connect',
        sessionId,
        context: this.context,
      },
      '*',
      [channel.port2],
    );

    try {
      await ready;
    } catch (cause) {
      if (generation !== this.generation) throw cause;
      const message = errorMessage(cause, 'Surface bridge handshake failed');
      this.fail(message);
      throw new Error(message);
    } finally {
      this.clearHandshake();
    }
    if (generation !== this.generation || !this.port) {
      throw new Error('Surface bridge connection was superseded');
    }
    this.emit({ status: 'healthy' });
    this.startHeartbeat();
    if (this.inspectionEnabled) this.sendInspectionMode();
  }

  setInspectionMode(enabled: boolean): void {
    this.inspectionEnabled = enabled;
    this.sendInspectionMode();
  }

  private sendInspectionMode(): void {
    if (!this.port || !this.sessionId || this.status !== 'healthy') return;
    this.port.postMessage({
      protocol: PROTOCOL,
      type: 'inspection',
      sessionId: this.sessionId,
      enabled: this.inspectionEnabled,
    });
  }

  async refresh(): Promise<void> {
    if (this.status !== 'healthy' || !this.port || !this.sessionId) return;
    try {
      this.context = await this.loadContext();
      this.port.postMessage({
        protocol: PROTOCOL,
        type: 'context',
        sessionId: this.sessionId,
        context: this.context,
      });
    } catch (cause) {
      const message = errorMessage(cause, 'Surface context refresh timed out');
      this.fail(message);
      throw new Error(message);
    }
  }

  disconnect(notify = true): void {
    this.generation += 1;
    this.teardown(new Error('Surface bridge disconnected'));
    if (notify) this.emit({ status: 'disconnected' });
  }

  private emit(update: SurfaceBridgeStatusUpdate): void {
    this.status = update.status;
    this.options.onStatus?.(update);
  }

  private async loadContext(): Promise<SurfaceRuntimeContext> {
    this.contextController?.abort();
    if (this.contextTimeout) clearTimeout(this.contextTimeout);
    const controller = new AbortController();
    this.contextController = controller;
    const timeout = setTimeout(
      () => controller.abort(),
      this.options.contextTimeoutMs,
    );
    this.contextTimeout = timeout;
    try {
      return await this.dependencies.getContext(
        this.eventId,
        this.buildId,
        controller.signal,
      );
    } finally {
      if (this.contextController === controller) this.contextController = null;
      clearTimeout(timeout);
      if (this.contextTimeout === timeout) this.contextTimeout = null;
    }
  }

  private clearHandshake(): void {
    if (this.handshakeTimeout) clearTimeout(this.handshakeTimeout);
    this.handshakeTimeout = null;
    this.readyResolve = null;
    this.readyReject = null;
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) clearInterval(this.heartbeatTimer);
    this.heartbeatTimer = null;
    this.pendingPing = null;
  }

  private startHeartbeat(): void {
    this.stopHeartbeat();
    this.sendHeartbeat();
    this.heartbeatTimer = setInterval(
      () => this.sendHeartbeat(),
      this.options.heartbeatIntervalMs,
    );
  }

  private sendHeartbeat(): void {
    if (!this.port || !this.sessionId || this.status !== 'healthy') return;
    const now = this.dependencies.now();
    if (this.pendingPing) {
      if (now - this.pendingPing.sentAt >= this.options.heartbeatGraceMs) {
        this.fail('Surface stopped responding to Aloy');
      }
      return;
    }
    const nonce = this.dependencies.randomId();
    this.pendingPing = { nonce, sentAt: now };
    this.port.postMessage({
      protocol: PROTOCOL,
      type: 'ping',
      sessionId: this.sessionId,
      nonce,
    });
  }

  private fail(message: string): void {
    if (this.port && this.sessionId) {
      this.port.postMessage({
        protocol: PROTOCOL,
        type: 'runtime',
        sessionId: this.sessionId,
        status: 'degraded',
        message,
      });
    }
    this.teardown(new Error(message));
    this.emit({ status: 'degraded', message });
  }

  private teardown(error: Error): void {
    this.contextController?.abort();
    this.contextController = null;
    if (this.contextTimeout) clearTimeout(this.contextTimeout);
    this.contextTimeout = null;
    const rejectReady = this.readyReject;
    this.clearHandshake();
    this.stopHeartbeat();
    for (const request of this.inFlight.values()) {
      clearTimeout(request.timeout);
      request.controller.abort();
    }
    this.inFlight.clear();
    rejectReady?.(error);
    this.port?.close();
    this.port = null;
    this.context = null;
    this.sessionId = null;
  }

  private respond(
    requestId: string,
    response:
      | { ok: true; result: unknown }
      | {
          ok: false;
          error: string;
          errorCode: SurfaceBridgeErrorCode;
          serverCode?: string;
          attemptId?: string;
          statusCode?: number;
          retryable?: boolean;
        },
  ): void {
    if (!this.port || !this.sessionId) return;
    this.port.postMessage({
      protocol: PROTOCOL,
      type: 'response',
      sessionId: this.sessionId,
      requestId,
      ...response,
    });
  }

  private async handle(value: unknown, generation: number): Promise<void> {
    if (generation !== this.generation || !this.port || !this.context) return;
    const message = value as {
      protocol?: string;
      type?: string;
      sessionId?: string;
      nonce?: string;
      selection?: unknown;
    } | null;
    if (
      !message
      || message.protocol !== PROTOCOL
      || message.sessionId !== this.sessionId
      || !message.type
    ) return;
    if (message.type === 'ready') {
      this.readyResolve?.();
      return;
    }
    if (message.type === 'pong') {
      if (message.nonce && message.nonce === this.pendingPing?.nonce) {
        this.pendingPing = null;
      }
      return;
    }
    if (message.type === 'selection') {
      try {
        const raw = object(message.selection);
        const bounds = object(raw.bounds);
        const styles = object(raw.styles);
        const finite = (value: unknown) => {
          if (typeof value !== 'number' || !Number.isFinite(value)) {
            throw new Error('Surface selection bounds are invalid');
          }
          return Math.round(Math.max(-100_000, Math.min(100_000, value)));
        };
        const nullableString = (value: unknown, name: string, max: number) => (
          value === null || value === undefined ? null : string(value, name, max)
        );
        const selection: SurfaceElementSelection = {
          selectionId: string(raw.selectionId, 'selection id', 200),
          buildId: this.context.build_id,
          codeRevisionId: this.context.code_revision_id,
          nodeId: string(raw.nodeId, 'node id', 300),
          tagName: string(raw.tagName, 'tag name', 50),
          role: string(raw.role, 'role', 80),
          accessibleName: typeof raw.accessibleName === 'string'
            ? raw.accessibleName.trim().slice(0, 300)
            : '',
          text: typeof raw.text === 'string' ? raw.text.trim().slice(0, 1_000) : '',
          componentId: string(raw.componentId, 'component id', 200),
          resource: nullableString(raw.resource, 'resource', 100),
          source: nullableString(raw.source, 'source', 300),
          bounds: {
            x: finite(bounds.x),
            y: finite(bounds.y),
            width: Math.max(0, finite(bounds.width)),
            height: Math.max(0, finite(bounds.height)),
          },
          styles: {
            display: string(styles.display, 'display style', 80),
            color: string(styles.color, 'color style', 100),
            backgroundColor: string(styles.backgroundColor, 'background style', 100),
            fontSize: string(styles.fontSize, 'font size', 50),
          },
        };
        this.inspectionEnabled = false;
        this.options.onElementSelection?.(selection);
      } catch {
        // Selection context is advisory and grants no authority. Malformed
        // iframe metadata is discarded without degrading normal Surface use.
      }
      return;
    }
    if (message.type !== 'request') return;
    await this.handleRequest(message as BridgeRequest);
  }

  private async handleRequest(request: BridgeRequest): Promise<void> {
    if (
      typeof request.requestId !== 'string'
      || request.requestId.length > 200
      || !request.method
    ) return;
    if (this.inFlight.size >= MAX_IN_FLIGHT) {
      this.respond(request.requestId, {
        ok: false,
        error: 'Too many Surface requests',
        errorCode: 'rate_limited',
      });
      return;
    }
    if (this.inFlight.has(request.requestId)) return;
    if (request.method === 'getContext') {
      this.respond(request.requestId, { ok: true, result: this.context });
      return;
    }
    if (request.method === 'openResource') {
      try {
        const params = object(request.params);
        const fileId = string(params.fileId, 'file id', 200);
        const componentId = typeof params.componentId === 'string'
          ? params.componentId.trim().slice(0, 200) || 'surface'
          : 'surface';
        if (!this.context?.capabilities.includes('files')) {
          this.respond(request.requestId, {
            ok: false,
            error: 'This Surface does not have access to Event files',
            errorCode: 'permission_denied',
            retryable: false,
          });
          return;
        }
        const file = this.context.data.files?.find((candidate) => candidate.id === fileId);
        if (!file) {
          this.respond(request.requestId, {
            ok: false,
            error: 'This resource is no longer available in the Event',
            errorCode: 'invalid',
            retryable: false,
          });
          return;
        }
        if (!this.options.onOpenResource) {
          this.respond(request.requestId, {
            ok: false,
            error: 'The Event resource viewer is unavailable',
            errorCode: 'unavailable',
            retryable: true,
          });
          return;
        }
        await this.options.onOpenResource({ fileId, componentId });
        this.respond(request.requestId, {
          ok: true,
          result: { opened: true, fileId },
        });
      } catch (cause) {
        this.respond(request.requestId, {
          ok: false,
          error: errorMessage(cause, 'Could not open this Event resource'),
          errorCode: 'failed',
          retryable: false,
        });
      }
      return;
    }

    const controller = new AbortController();
    const timeout = setTimeout(
      () => controller.abort(),
      this.options.requestTimeoutMs,
    );
    this.inFlight.set(request.requestId, { controller, timeout });
    let result: SurfaceInteractionResponse;
    let handoff: Omit<SurfaceAloyHandoff, 'response'> | null = null;
    try {
      const params = object(request.params);
      const componentId =
        typeof params.componentId === 'string' ? params.componentId : 'surface';
      const idempotencyKey = string(
        params.idempotencyKey,
        'idempotency key',
        200,
      );
      let method: SurfaceInteractionMethod;
      let name: string;
      let bodyPayload: Record<string, unknown>;
      let message: string | undefined;
      let reason: string | undefined;
      let refs: SurfaceResourceRef[] = [];
      if (request.method === 'command' || request.method === 'dispatch') {
        method = request.method;
        name = string(params.name, 'intent name', 128);
        bodyPayload = payload(params.payload);
      } else if (request.method === 'askAloy') {
        method = 'ask_aloy';
        name = 'aloy.ask';
        message = string(params.message, 'message');
        bodyPayload = payload(params.context);
        refs = resourceRefs(params.resources);
      } else if (request.method === 'requestAction') {
        method = 'request_action';
        const action = object(params.action);
        name = string(action.name, 'action name', 128);
        bodyPayload = payload(action.payload);
        reason =
          typeof action.reason === 'string'
            ? action.reason.slice(0, 4000)
            : undefined;
      } else {
        throw new Error('Unsupported Surface bridge method');
      }
      handoff = {
        method,
        name,
        componentId: componentId.slice(0, 200),
        message,
        reason,
        resourceRefs: refs,
      };
      if (!this.context) throw new Error('Surface context is unavailable');
      if (refs.length) {
        if (!this.context.capabilities.includes('files')) {
          throw new Error('This Surface does not have access to Event files');
        }
        const available = new Set(
          (this.context.data.files ?? []).map((file) => file.id),
        );
        if (refs.some((ref) => !available.has(ref.id))) {
          throw new Error('One or more Event resources are no longer available');
        }
      }
      result = await this.dependencies.createInteraction(
        this.eventId,
        {
          build_id: this.context.build_id,
          code_revision_id: this.context.code_revision_id,
          data_revision: this.context.data_revision,
          method,
          name,
          component_id: componentId.slice(0, 200),
          payload: bodyPayload,
          message,
          reason,
          resource_refs: refs,
          idempotency_key: idempotencyKey,
        },
        controller.signal,
      );
    } catch (cause) {
      const details = apiErrorDetails(cause);
      const hasDurableAttempt = typeof details.attempt_id === 'string';
      if (
        cause instanceof ApiError
        && (cause.status === 409 || hasDurableAttempt)
        && this.port
      ) {
        try {
          this.context = await this.loadContext();
          this.port.postMessage({
            protocol: PROTOCOL,
            type: 'context',
            sessionId: this.sessionId,
            context: this.context,
          });
        } catch {
          // The original conflict remains authoritative. A reconnect or
          // publication change is reported by the normal runtime lifecycle.
        }
      }
      this.respond(request.requestId, {
        ok: false,
        error: errorMessage(cause, 'Surface request timed out'),
        errorCode: failureCode(cause, controller),
        serverCode:
          typeof details.code === 'string'
            ? String(details.code)
            : undefined,
        attemptId:
          typeof details.attempt_id === 'string'
            ? String(details.attempt_id)
            : undefined,
        statusCode: cause instanceof ApiError ? cause.status : undefined,
        retryable: retryableFailure(cause, controller),
      });
      return;
    } finally {
      clearTimeout(timeout);
      this.inFlight.delete(request.requestId);
    }

    if (result.data_revision !== null || result.proposal_id || result.handling_run_id) {
      try {
        await this.refresh();
      } catch {
        // The interaction is durable, but the generated UI must not claim a
        // reconciled outcome from stale context. Keep its request pending so
        // the normal reconnect can replay the same idempotency key safely.
        return;
      }
    }
    if (handoff && shouldSummonAloy(handoff.method, result)) {
      try {
        this.options.onAloyHandoff?.({ ...handoff, response: result });
      } catch {
        // Host chrome is an enhancement around a completed durable action. A
        // rendering failure must never hide the successful result from the
        // Surface or invite an unsafe retry.
      }
    }
    // MessageChannel preserves sender order: canonical context is delivered
    // before this acknowledgement, so a resolved SDK Promise never races the
    // refreshed Event data on the normal path.
    this.respond(request.requestId, { ok: true, result });
  }
}
