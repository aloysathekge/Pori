import { afterEach, describe, expect, mock, test } from 'bun:test';
import {
  installEventTemplate,
  listEventTemplates,
} from '../src/api/eventTemplates';

const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
});

describe('Event template API', () => {
  test('reads the published discovery catalog', async () => {
    const fetchMock = mock(async () => new Response(JSON.stringify({
      templates: [{
        id: 'template-career-os',
        slug: 'career-os',
        title: 'Career OS',
        summary: 'Manage a focused job search.',
        discovery_group: 'professional',
        current_release: { id: 'release-career-os-v1', version: 1, schema_version: 1 },
        updated_at: '2026-07-22T00:00:00Z',
      }],
    }), { status: 200, headers: { 'Content-Type': 'application/json' } }));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const response = await listEventTemplates();

    expect(response.templates[0]?.slug).toBe('career-os');
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(String(fetchMock.mock.calls[0]?.[0])).toBe('http://localhost:8000/v1/event-templates');
  });

  test('pins installation to the reviewed release and stable request key', async () => {
    const fetchMock = mock(async () => new Response(JSON.stringify({
      installation: {
        id: 'installation-1',
        template_id: 'template-career-os',
        release_id: 'release-career-os-v1',
        event_id: 'event-1',
        status: 'installed',
        installed_at: '2026-07-22T00:00:00Z',
      },
      event: { id: 'event-1', title: 'My Career OS' },
      surface: {
        project_id: 'surface-project-1',
        status: 'preparing',
        run_id: 'run-1',
        run_status: 'pending',
      },
      replayed: false,
    }), { status: 201, headers: { 'Content-Type': 'application/json' } }));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const result = await installEventTemplate('template-career-os', {
      idempotency_key: 'event-template:template-career-os:request-1',
      release_id: 'release-career-os-v1',
      title: 'My Career OS',
    });

    expect(result.surface.status).toBe('preparing');
    const request = fetchMock.mock.calls[0]?.[1] as RequestInit;
    expect(request.method).toBe('POST');
    expect(JSON.parse(String(request.body))).toEqual({
      idempotency_key: 'event-template:template-career-os:request-1',
      release_id: 'release-career-os-v1',
      title: 'My Career OS',
    });
  });
});
