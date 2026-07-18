import { describe, expect, test } from 'bun:test';
import type {
  SurfaceInteractionResponse,
  SurfaceRuntimeContext,
} from '../src/api/surfaces';
import { ApiError } from '../src/api/client';
import { SurfaceBridgeHost } from '../src/components/surfaces/surfaceBridge';

const context: SurfaceRuntimeContext = {
  protocol_version: '1',
  command_contract_version: '1',
  sdk_version: '1',
  event_id: 'event-1',
  project_id: 'project-1',
  build_id: 'build-1',
  code_revision_id: 'revision-1',
  data_revision: 1,
  capabilities: [],
  widgets: [],
  data: {},
};

const interaction: SurfaceInteractionResponse = {
  id: 'interaction-1',
  status: 'committed',
  name: 'trip.select_flight',
  interaction_class: 'intent',
  data_revision: 2,
  handling_run_id: null,
  proposal_id: null,
  request_message_id: null,
  outcome_message_id: null,
  result: {},
  replayed: false,
};

type ConnectMessage = {
  protocol: '1';
  type: 'aloy.surface.connect';
  sessionId: string;
};

function frame(
  onConnect: (message: ConnectMessage, port: MessagePort) => void,
): HTMLIFrameElement {
  return {
    contentWindow: {
      postMessage(message: ConnectMessage, _target: string, ports: MessagePort[]) {
        onConnect(message, ports[0]);
      },
    },
  } as unknown as HTMLIFrameElement;
}

function responsivePort(message: ConnectMessage, port: MessagePort) {
  port.onmessage = (event: MessageEvent<{ type?: string; nonce?: string }>) => {
    if (event.data.type === 'ping') {
      port.postMessage({
        protocol: '1',
        type: 'pong',
        sessionId: message.sessionId,
        nonce: event.data.nonce,
      });
    }
  };
  port.start();
  port.postMessage({ protocol: '1', type: 'ready', sessionId: message.sessionId });
}

describe('SurfaceBridgeHost', () => {
  test('becomes healthy only after an acknowledgement for the bound session', async () => {
    const statuses: string[] = [];
    const bridge = new SurfaceBridgeHost(
      'event-1',
      'build-1',
      { onStatus: ({ status }) => statuses.push(status) },
      { getContext: async () => context },
    );

    await bridge.connect(frame(responsivePort));

    expect(bridge.currentStatus).toBe('healthy');
    expect(statuses).toEqual(['connecting', 'healthy']);
    bridge.disconnect(false);
  });

  test('rejects an acknowledgement from a different session', async () => {
    const bridge = new SurfaceBridgeHost(
      'event-1',
      'build-1',
      { handshakeTimeoutMs: 15 },
      { getContext: async () => context },
    );
    const wrongSession = frame((_message, port) => {
      port.start();
      port.postMessage({ protocol: '1', type: 'ready', sessionId: 'wrong-session' });
    });

    await expect(bridge.connect(wrongSession)).rejects.toThrow(
      'Surface did not acknowledge the secure bridge',
    );
    expect(bridge.currentStatus).toBe('degraded');
  });

  test('degrades when the acknowledged runtime stops answering heartbeats', async () => {
    let clock = 0;
    const bridge = new SurfaceBridgeHost(
      'event-1',
      'build-1',
      { heartbeatIntervalMs: 5, heartbeatGraceMs: 5 },
      {
        getContext: async () => context,
        now: () => {
          clock += 5;
          return clock;
        },
      },
    );
    await bridge.connect(
      frame((message, port) => {
        port.start();
        port.postMessage({
          protocol: '1',
          type: 'ready',
          sessionId: message.sessionId,
        });
      }),
    );

    await new Promise((resolve) => setTimeout(resolve, 30));
    expect(bridge.currentStatus).toBe('degraded');
  });

  test('routes a session-bound interaction with its idempotency key', async () => {
    let surfacePort: MessagePort | null = null;
    let sessionId = '';
    let receivedIdempotencyKey = '';
    let receivedMethod = '';
    let contextReads = 0;
    const receivedTypes: string[] = [];
    const bridge = new SurfaceBridgeHost(
      'event-1',
      'build-1',
      {},
      {
        getContext: async () => ({
          ...context,
          data_revision: ++contextReads,
        }),
        createInteraction: async (_eventId, request) => {
          receivedIdempotencyKey = request.idempotency_key;
          receivedMethod = request.method;
          return interaction;
        },
      },
    );
    await bridge.connect(
      frame((message, port) => {
        surfacePort = port;
        sessionId = message.sessionId;
        responsivePort(message, port);
      }),
    );
    await new Promise((resolve) => setTimeout(resolve, 0));
    const response = new Promise<Record<string, unknown>>((resolve, reject) => {
      if (!surfacePort) throw new Error('Surface port is unavailable');
      surfacePort.onmessage = (event: MessageEvent<Record<string, unknown>>) => {
        if (typeof event.data.type === 'string') receivedTypes.push(event.data.type);
        if (event.data.type === 'ping') {
          surfacePort?.postMessage({
            protocol: '1',
            type: 'pong',
            sessionId,
            nonce: event.data.nonce,
          });
        } else if (event.data.type === 'response') {
          resolve(event.data);
        }
      };
      surfacePort.postMessage({
        protocol: '1',
        type: 'request',
        sessionId,
        requestId: 'request-1',
        method: 'command',
        params: {
          name: 'trip.select_flight',
          payload: {},
          componentId: 'flight-card',
          idempotencyKey: 'interaction-key-1',
        },
      });
      setTimeout(() => reject(new Error('Timed out waiting for host response')), 500);
    });

    await expect(response).resolves.toMatchObject({
      ok: true,
      requestId: 'request-1',
    });
    expect(receivedIdempotencyKey).toBe('interaction-key-1');
    expect(receivedMethod).toBe('command');
    expect(receivedTypes.indexOf('context')).toBeGreaterThan(-1);
    expect(receivedTypes.indexOf('context')).toBeLessThan(
      receivedTypes.indexOf('response'),
    );
    expect(contextReads).toBe(2);
    bridge.disconnect(false);
  });

  test('returns a structured conflict after refreshing canonical context', async () => {
    let surfacePort: MessagePort | null = null;
    let sessionId = '';
    let contextReads = 0;
    const receivedTypes: string[] = [];
    const bridge = new SurfaceBridgeHost(
      'event-1',
      'build-1',
      {},
      {
        getContext: async () => ({
          ...context,
          data_revision: ++contextReads,
        }),
        createInteraction: async () => {
          throw new ApiError(409, 'Surface data revision changed');
        },
      },
    );
    await bridge.connect(
      frame((message, port) => {
        surfacePort = port;
        sessionId = message.sessionId;
        responsivePort(message, port);
      }),
    );
    await new Promise((resolve) => setTimeout(resolve, 0));
    const response = new Promise<Record<string, unknown>>((resolve, reject) => {
      if (!surfacePort) throw new Error('Surface port is unavailable');
      surfacePort.onmessage = (event: MessageEvent<Record<string, unknown>>) => {
        if (typeof event.data.type === 'string') receivedTypes.push(event.data.type);
        if (event.data.type === 'ping') {
          surfacePort?.postMessage({
            protocol: '1',
            type: 'pong',
            sessionId,
            nonce: event.data.nonce,
          });
        } else if (event.data.type === 'response') {
          resolve(event.data);
        }
      };
      surfacePort.postMessage({
        protocol: '1',
        type: 'request',
        sessionId,
        requestId: 'request-conflict',
        method: 'command',
        params: {
          name: 'trip.select_flight',
          payload: {},
          componentId: 'flight-card',
          idempotencyKey: 'interaction-conflict-1',
        },
      });
      setTimeout(() => reject(new Error('Timed out waiting for host response')), 500);
    });

    await expect(response).resolves.toMatchObject({
      ok: false,
      errorCode: 'conflict',
      statusCode: 409,
      retryable: true,
    });
    expect(receivedTypes.indexOf('context')).toBeLessThan(
      receivedTypes.indexOf('response'),
    );
    expect(contextReads).toBe(2);
    bridge.disconnect(false);
  });
});
