import { describe, expect, test } from 'bun:test';
import { resolveFileVisual } from '../src/components/files/fileVisualTypes';

describe('resolveFileVisual', () => {
  test('uses file-specific render identities', () => {
    expect(resolveFileVisual({ name: 'paper.pdf' })).toBe('pdf');
    expect(resolveFileVisual({ name: 'essay.docx' })).toBe('document');
    expect(resolveFileVisual({ name: 'marks.xlsx' })).toBe('spreadsheet');
    expect(resolveFileVisual({ name: 'lecture.pptx' })).toBe('slides');
    expect(resolveFileVisual({ name: 'photo.webp' })).toBe('image');
    expect(resolveFileVisual({ name: 'recording.mp4' })).toBe('video');
    expect(resolveFileVisual({ name: 'interview.m4a' })).toBe('audio');
    expect(resolveFileVisual({ name: 'source.tsx' })).toBe('code');
    expect(resolveFileVisual({ name: 'bundle.zip' })).toBe('archive');
  });

  test('uses MIME when a file has no useful extension', () => {
    expect(resolveFileVisual({ name: 'upload', content_type: 'video/webm' })).toBe('video');
    expect(resolveFileVisual({ name: 'scan', content_type: 'application/pdf' })).toBe('pdf');
  });

  test('keeps an unknown run output visibly identifiable as an artifact', () => {
    expect(resolveFileVisual({ name: 'result', kind: 'artifact' })).toBe('artifact');
  });
});
