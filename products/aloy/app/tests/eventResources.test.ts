import { describe, expect, test } from 'bun:test';
import { groupEventResources } from '../src/components/events/eventResources';
import type { EventFile } from '../src/api/events';

function eventFile(id: string, kind: string): EventFile {
  return {
    id,
    name: `${id}.md`,
    kind,
    content_type: 'text/markdown',
    size_bytes: 100,
    origin_session_id: null,
    origin_run_id: kind === 'artifact' ? 'run-1' : null,
    created_at: '2026-07-22T00:00:00Z',
  };
}

describe('Event Resources', () => {
  test('keeps trusted inputs separate from Aloy-generated artifacts', () => {
    const groups = groupEventResources([
      eventFile('course-outline', 'upload'),
      eventFile('weekly-plan', 'artifact'),
    ]);

    expect(groups.sources.map((file) => file.id)).toEqual(['course-outline']);
    expect(groups.artifacts.map((file) => file.id)).toEqual(['weekly-plan']);
  });

  test('keeps future non-artifact input kinds with Sources by default', () => {
    const groups = groupEventResources([eventFile('saved-link-export', 'source')]);

    expect(groups.sources.map((file) => file.id)).toEqual(['saved-link-export']);
    expect(groups.artifacts).toEqual([]);
  });
});
