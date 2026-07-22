import { describe, expect, test } from 'bun:test';
import type {
  SurfaceInteractionResponse,
  SurfaceRuntimeContext,
} from '../src/api/surfaces';
import { ApiError } from '../src/api/client';
import {
  SurfaceBridgeHost,
  shouldSummonAloy,
} from '../src/components/surfaces/surfaceBridge';

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
  test('summons host Aloy only for reasoning, protected action, Run, or Proposal handoffs', () => {
    expect(shouldSummonAloy('command', interaction)).toBe(false);
    expect(shouldSummonAloy('dispatch', interaction)).toBe(false);
    expect(shouldSummonAloy('ask_aloy', interaction)).toBe(true);
    expect(shouldSummonAloy('request_action', interaction)).toBe(true);
    expect(shouldSummonAloy('command', { ...interaction, handling_run_id: 'run-1' })).toBe(true);
    expect(shouldSummonAloy('command', { ...interaction, proposal_id: 'proposal-1' })).toBe(true);
  });

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

  test('binds an inspected element to the authenticated build and revision', async () => {
    let surfacePort: MessagePort | null = null;
    let sessionId = '';
    const selections: Array<Record<string, unknown>> = [];
    const bridge = new SurfaceBridgeHost(
      'event-1',
      'build-1',
      { onElementSelection: (selection) => selections.push(selection) },
      { getContext: async () => context },
    );
    await bridge.connect(
      frame((message, port) => {
        surfacePort = port;
        sessionId = message.sessionId;
        responsivePort(message, port);
      }),
    );
    bridge.setInspectionMode(true);
    await new Promise((resolve) => setTimeout(resolve, 0));
    if (!surfacePort) throw new Error('Surface port is unavailable');
    surfacePort.postMessage({
      protocol: '1',
      type: 'selection',
      sessionId,
      selection: {
        selectionId: 'selection-1',
        nodeId: 'main:0/button:0',
        tagName: 'button',
        role: 'button',
        accessibleName: 'Save application',
        text: 'Save application',
        componentId: 'application-form',
        resource: 'data:career',
        source: '/src/App.tsx:42:8',
        bounds: { x: 10, y: 20, width: 180, height: 44 },
        styles: {
          display: 'inline-flex',
          color: 'rgb(255, 255, 255)',
          backgroundColor: 'rgb(15, 133, 113)',
          fontSize: '14px',
        },
        // The iframe cannot choose authority; these are intentionally ignored.
        buildId: 'forged-build',
        codeRevisionId: 'forged-revision',
      },
    });
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(selections).toHaveLength(1);
    expect(selections[0]).toMatchObject({
      buildId: 'build-1',
      codeRevisionId: 'revision-1',
      accessibleName: 'Save application',
      source: '/src/App.tsx:42:8',
    });
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

  test('opens a declared Event file through host chrome without creating an interaction', async () => {
    let surfacePort: MessagePort | null = null;
    let sessionId = '';
    let interactionCalls = 0;
    const opened: Array<{ fileId: string; componentId: string }> = [];
    const fileContext: SurfaceRuntimeContext = {
      ...context,
      capabilities: ['files'],
      data: {
        files: [{
          id: 'file-1',
          name: 'semester-plan.md',
          kind: 'artifact',
          content_type: 'text/markdown',
          size_bytes: 2048,
          origin_session_id: 'conversation-1',
          origin_run_id: 'run-1',
          created_at: '2026-07-22T12:00:00Z',
        }],
      },
    };
    const bridge = new SurfaceBridgeHost(
      'event-1',
      'build-1',
      { onOpenResource: (request) => { opened.push(request); } },
      {
        getContext: async () => fileContext,
        createInteraction: async () => {
          interactionCalls += 1;
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
        requestId: 'request-open-file',
        method: 'openResource',
        params: { fileId: 'file-1', componentId: 'reading-list' },
      });
      setTimeout(() => reject(new Error('Timed out waiting for host response')), 500);
    });

    await expect(response).resolves.toMatchObject({
      ok: true,
      result: { opened: true, fileId: 'file-1' },
    });
    expect(opened).toEqual([{ fileId: 'file-1', componentId: 'reading-list' }]);
    expect(interactionCalls).toBe(0);
    bridge.disconnect(false);
  });

  test('refuses resource ids outside the capability-scoped Event context', async () => {
    let surfacePort: MessagePort | null = null;
    let sessionId = '';
    let opened = false;
    const bridge = new SurfaceBridgeHost(
      'event-1',
      'build-1',
      { onOpenResource: () => { opened = true; } },
      {
        getContext: async () => ({
          ...context,
          capabilities: ['files'],
          data: { files: [] },
        }),
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
        if (event.data.type === 'response') resolve(event.data);
      };
      surfacePort.postMessage({
        protocol: '1',
        type: 'request',
        sessionId,
        requestId: 'request-open-missing-file',
        method: 'openResource',
        params: { fileId: 'other-event-file' },
      });
      setTimeout(() => reject(new Error('Timed out waiting for host response')), 500);
    });

    await expect(response).resolves.toMatchObject({
      ok: false,
      errorCode: 'invalid',
      retryable: false,
    });
    expect(opened).toBe(false);
    bridge.disconnect(false);
  });

  test('notifies host chrome without letting its failure hide a successful handoff', async () => {
    let surfacePort: MessagePort | null = null;
    let sessionId = '';
    const handoffs: Array<{ method: string; name: string; message?: string }> = [];
    let forwardedResourceIds: string[] = [];
    const bridge = new SurfaceBridgeHost(
      'event-1',
      'build-1',
      {
        onAloyHandoff: (handoff) => {
          handoffs.push(handoff);
          throw new Error('host panel failed to render');
        },
      },
      {
        getContext: async () => ({
          ...context,
          capabilities: ['ask_aloy', 'files'],
          data: {
            files: [{
              id: 'file-brief',
              name: 'flight-brief.md',
              kind: 'artifact',
              content_type: 'text/markdown',
              size_bytes: 400,
              origin_session_id: 'conversation-1',
              origin_run_id: 'run-brief',
              created_at: '2026-07-22T12:00:00Z',
            }],
          },
        }),
        createInteraction: async (_eventId, request) => {
          forwardedResourceIds = (request.resource_refs ?? []).map((ref) => ref.id);
          return {
            ...interaction,
            name: 'aloy.ask',
            interaction_class: 'reasoning',
            data_revision: null,
            handling_run_id: 'run-ask-1',
          };
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
        requestId: 'request-ask-aloy',
        method: 'askAloy',
        params: {
          message: 'Compare the selected flights',
          context: { selectedFlightId: 'flight-2' },
          resources: [{ type: 'file', id: 'file-brief' }],
          componentId: 'flight-comparison',
          idempotencyKey: 'interaction-ask-1',
        },
      });
      setTimeout(() => reject(new Error('Timed out waiting for host response')), 500);
    });

    await expect(response).resolves.toMatchObject({ ok: true });
    expect(handoffs).toHaveLength(1);
    expect(handoffs[0]).toMatchObject({
      method: 'ask_aloy',
      name: 'aloy.ask',
      message: 'Compare the selected flights',
      resourceRefs: [{ type: 'file', id: 'file-brief' }],
    });
    expect(forwardedResourceIds).toEqual(['file-brief']);
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
          throw new ApiError(409, 'Surface data revision changed', {
            code: 'stale_data_revision',
            retryable: true,
            attempt_id: 'scat-conflict-1',
          });
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
      serverCode: 'stale_data_revision',
      attemptId: 'scat-conflict-1',
      statusCode: 409,
      retryable: true,
    });
    expect(receivedTypes.indexOf('context')).toBeLessThan(
      receivedTypes.indexOf('response'),
    );
    expect(contextReads).toBe(2);
    bridge.disconnect(false);
  });

  test('returns a durable permission rejection without retrying it', async () => {
    let surfacePort: MessagePort | null = null;
    let sessionId = '';
    let calls = 0;
    const bridge = new SurfaceBridgeHost(
      'event-1',
      'build-1',
      {},
      {
        getContext: async () => context,
        createInteraction: async () => {
          calls += 1;
          throw new ApiError(403, 'Surface action tool is denied by policy', {
            code: 'permission_denied',
            retryable: false,
            attempt_id: 'scat-permission-1',
          });
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
        requestId: 'request-permission',
        method: 'requestAction',
        params: {
          action: { name: 'career.send_email', payload: {} },
          componentId: 'send-email',
          idempotencyKey: 'interaction-permission-1',
        },
      });
      setTimeout(() => reject(new Error('Timed out waiting for host response')), 500);
    });

    await expect(response).resolves.toMatchObject({
      ok: false,
      errorCode: 'permission_denied',
      serverCode: 'permission_denied',
      attemptId: 'scat-permission-1',
      statusCode: 403,
      retryable: false,
    });
    expect(calls).toBe(1);
    bridge.disconnect(false);
  });

  test('withholds success until canonical refresh recovers on reconnect', async () => {
    let contextReads = 0;
    let interactionCalls = 0;
    let surfacePort: MessagePort | null = null;
    let sessionId = '';
    const statuses: string[] = [];
    const bridge = new SurfaceBridgeHost(
      'event-1',
      'build-1',
      { onStatus: ({ status }) => statuses.push(status) },
      {
        getContext: async () => {
          contextReads += 1;
          if (contextReads === 2) throw new TypeError('refresh unavailable');
          return { ...context, data_revision: contextReads };
        },
        createInteraction: async () => {
          interactionCalls += 1;
          return { ...interaction, replayed: interactionCalls > 1 };
        },
      },
    );
    const runtimeFrame = frame((message, port) => {
      surfacePort = port;
      sessionId = message.sessionId;
      responsivePort(message, port);
    });
    await bridge.connect(runtimeFrame);
    let falseAcknowledgement = false;
    if (!surfacePort) throw new Error('Surface port is unavailable');
    surfacePort.onmessage = (event: MessageEvent<Record<string, unknown>>) => {
      if (event.data.type === 'ping') {
        surfacePort?.postMessage({
          protocol: '1',
          type: 'pong',
          sessionId,
          nonce: event.data.nonce,
        });
      } else if (event.data.type === 'response') {
        falseAcknowledgement = true;
      }
    };
    surfacePort.postMessage({
      protocol: '1',
      type: 'request',
      sessionId,
      requestId: 'request-before-reconnect',
      method: 'command',
      params: {
        name: 'career.create',
        payload: { id: 'app-1' },
        componentId: 'add-application',
        idempotencyKey: 'interaction-reconnect-1',
      },
    });
    await new Promise((resolve) => setTimeout(resolve, 30));
    expect(falseAcknowledgement).toBe(false);
    expect(bridge.currentStatus).toBe('degraded');

    await bridge.connect(runtimeFrame);
    await new Promise((resolve) => setTimeout(resolve, 0));
    const recovered = new Promise<Record<string, unknown>>((resolve, reject) => {
      if (!surfacePort) throw new Error('Surface port is unavailable');
      surfacePort.onmessage = (event: MessageEvent<Record<string, unknown>>) => {
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
        requestId: 'request-after-reconnect',
        method: 'command',
        params: {
          name: 'career.create',
          payload: { id: 'app-1' },
          componentId: 'add-application',
          idempotencyKey: 'interaction-reconnect-1',
        },
      });
      setTimeout(() => reject(new Error('Timed out waiting for recovery')), 500);
    });

    await expect(recovered).resolves.toMatchObject({ ok: true });
    expect(interactionCalls).toBe(2);
    expect(statuses).toContain('degraded');
    expect(bridge.currentStatus).toBe('healthy');
    bridge.disconnect(false);
  });
});
