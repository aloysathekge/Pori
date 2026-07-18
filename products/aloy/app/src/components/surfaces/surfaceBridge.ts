import {
  createSurfaceInteraction,
  getSurfaceRuntimeContext,
  type SurfaceInteractionMethod,
  type SurfaceInteractionRequest,
  type SurfaceInteractionResponse,
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

export interface SurfaceBridgeStatusUpdate {
  status: SurfaceBridgeStatus;
  message?: string;
}

export interface SurfaceBridgeOptions {
  onStatus?: (update: SurfaceBridgeStatusUpdate) => void;
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
  method: 'getContext' | 'command' | 'dispatch' | 'askAloy' | 'requestAction';
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

function errorMessage(cause: unknown, fallback: string): string {
  if (cause instanceof Error && cause.name === 'AbortError') return fallback;
  return cause instanceof Error ? cause.message : fallback;
}

function retryableFailure(cause: unknown, controller: AbortController): boolean {
  return (
    controller.signal.aborted
    || cause instanceof TypeError
    || (cause instanceof ApiError && [409, 429, 503].includes(cause.status))
  );
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
  private readyResolve: (() => void) | null = null;
  private readyReject: ((error: Error) => void) | null = null;
  private readonly eventId: string;
  private readonly buildId: string;
  private readonly options: Required<Omit<SurfaceBridgeOptions, 'onStatus'>> &
    Pick<SurfaceBridgeOptions, 'onStatus'>;
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
      | { ok: false; error: string; retryable?: boolean },
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
      });
      return;
    }
    if (this.inFlight.has(request.requestId)) return;
    if (request.method === 'getContext') {
      this.respond(request.requestId, { ok: true, result: this.context });
      return;
    }

    const controller = new AbortController();
    const timeout = setTimeout(
      () => controller.abort(),
      this.options.requestTimeoutMs,
    );
    this.inFlight.set(request.requestId, { controller, timeout });
    let result: SurfaceInteractionResponse;
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
      if (request.method === 'command' || request.method === 'dispatch') {
        method = request.method;
        name = string(params.name, 'intent name', 128);
        bodyPayload = payload(params.payload);
      } else if (request.method === 'askAloy') {
        method = 'ask_aloy';
        name = 'aloy.ask';
        message = string(params.message, 'message');
        bodyPayload = payload(params.context);
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
      if (!this.context) throw new Error('Surface context is unavailable');
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
          idempotency_key: idempotencyKey,
        },
        controller.signal,
      );
    } catch (cause) {
      if (cause instanceof ApiError && cause.status === 409 && this.port) {
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
        retryable: retryableFailure(cause, controller),
      });
      return;
    } finally {
      clearTimeout(timeout);
      this.inFlight.delete(request.requestId);
    }

    this.respond(request.requestId, { ok: true, result });
    if (result.data_revision !== null || result.proposal_id || result.handling_run_id) {
      try {
        await this.refresh();
      } catch {
        // The interaction is already durable and acknowledged. Runtime status
        // reports the refresh failure; never send a contradictory second reply.
      }
    }
  }
}
