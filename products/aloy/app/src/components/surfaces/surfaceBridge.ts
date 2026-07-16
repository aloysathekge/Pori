import {
  createSurfaceInteraction,
  getSurfaceRuntimeContext,
  type SurfaceInteractionMethod,
  type SurfaceRuntimeContext,
} from '@/api/surfaces';

const PROTOCOL = '1' as const;
const MAX_IN_FLIGHT = 8;

type BridgeRequest = {
  protocol: '1';
  type: 'request';
  requestId: string;
  method: 'getContext' | 'dispatch' | 'askAloy' | 'requestAction';
  params?: unknown;
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

export class SurfaceBridgeHost {
  private port: MessagePort | null = null;
  private context: SurfaceRuntimeContext | null = null;
  private inFlight = new Set<string>();
  private readonly eventId: string;
  private readonly buildId: string;

  constructor(eventId: string, buildId: string) {
    this.eventId = eventId;
    this.buildId = buildId;
  }

  async connect(frame: HTMLIFrameElement): Promise<void> {
    this.disconnect();
    const frameWindow = frame.contentWindow;
    if (!frameWindow) throw new Error('Surface frame is unavailable');
    this.context = await getSurfaceRuntimeContext(this.eventId, this.buildId);
    const channel = new MessageChannel();
    this.port = channel.port1;
    this.port.onmessage = (event: MessageEvent<unknown>) => {
      void this.handle(event.data);
    };
    this.port.start();
    frameWindow.postMessage(
      {
        protocol: PROTOCOL,
        type: 'aloy.surface.connect',
        sessionId: crypto.randomUUID(),
        context: this.context,
      },
      '*',
      [channel.port2],
    );
  }

  async refresh(): Promise<void> {
    if (!this.port) return;
    this.context = await getSurfaceRuntimeContext(this.eventId, this.buildId);
    this.port.postMessage({
      protocol: PROTOCOL,
      type: 'context',
      context: this.context,
    });
  }

  disconnect(): void {
    this.port?.close();
    this.port = null;
    this.context = null;
    this.inFlight.clear();
  }

  private respond(
    requestId: string,
    response: { ok: true; result: unknown } | { ok: false; error: string },
  ) {
    this.port?.postMessage({
      protocol: PROTOCOL,
      type: 'response',
      requestId,
      ...response,
    });
  }

  private async handle(value: unknown): Promise<void> {
    if (!this.port || !this.context) return;
    const request = value as Partial<BridgeRequest> | null;
    if (
      !request
      || request.protocol !== PROTOCOL
      || request.type !== 'request'
      || typeof request.requestId !== 'string'
      || request.requestId.length > 200
      || !request.method
    ) return;
    if (this.inFlight.size >= MAX_IN_FLIGHT) {
      this.respond(request.requestId, { ok: false, error: 'Too many Surface requests' });
      return;
    }
    if (this.inFlight.has(request.requestId)) return;
    this.inFlight.add(request.requestId);
    try {
      if (request.method === 'getContext') {
        this.respond(request.requestId, { ok: true, result: this.context });
        return;
      }
      const params = object(request.params);
      const componentId = typeof params.componentId === 'string' ? params.componentId : 'surface';
      const idempotencyKey = string(params.idempotencyKey, 'idempotency key', 200);
      let method: SurfaceInteractionMethod;
      let name: string;
      let bodyPayload: Record<string, unknown>;
      let message: string | undefined;
      let reason: string | undefined;
      if (request.method === 'dispatch') {
        method = 'dispatch';
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
        reason = typeof action.reason === 'string' ? action.reason.slice(0, 4000) : undefined;
      } else {
        throw new Error('Unsupported Surface bridge method');
      }
      const result = await createSurfaceInteraction(this.eventId, {
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
      });
      this.respond(request.requestId, { ok: true, result });
      if (result.data_revision !== null || result.proposal_id || result.handling_run_id) {
        await this.refresh();
      }
    } catch (cause) {
      this.respond(request.requestId, {
        ok: false,
        error: cause instanceof Error ? cause.message : 'Surface request failed',
      });
    } finally {
      this.inFlight.delete(request.requestId);
    }
  }
}
