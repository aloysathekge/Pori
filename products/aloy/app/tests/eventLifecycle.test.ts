import { afterEach, describe, expect, mock, test } from 'bun:test';
import {
  listEvents,
  permanentlyDeleteEvent,
  updateEvent,
} from '../src/api/events';

const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
});

describe('Event lifecycle API', () => {
  test('lists archived Events explicitly', async () => {
    const fetchMock = mock(async () => new Response(JSON.stringify([]), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    }));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await listEvents('archived');

    expect(String(fetchMock.mock.calls[0]?.[0])).toBe(
      'http://localhost:8000/v1/events?lifecycle=archived',
    );
  });

  test('archives through the ordinary Event update boundary', async () => {
    const fetchMock = mock(async () => new Response(JSON.stringify({
      id: 'event-1',
      lifecycle: 'archived',
    }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    }));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await updateEvent('event-1', { lifecycle: 'archived' });

    const request = fetchMock.mock.calls[0]?.[1] as RequestInit;
    expect(request.method).toBe('PATCH');
    expect(JSON.parse(String(request.body))).toEqual({ lifecycle: 'archived' });
  });

  test('sends exact-name confirmation for permanent deletion', async () => {
    const fetchMock = mock(async () => new Response(JSON.stringify({
      deleted: true,
      event_id: 'event-1',
      storage_objects: 0,
      storage_cleanup: 'complete',
    }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    }));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await permanentlyDeleteEvent('event-1', 'Career OS');

    const request = fetchMock.mock.calls[0]?.[1] as RequestInit;
    expect(request.method).toBe('DELETE');
    expect(JSON.parse(String(request.body))).toEqual({ confirmation: 'Career OS' });
  });
});
