import { describe, expect, test } from 'bun:test';

import { SURFACE_IFRAME_SANDBOX } from '../src/components/surfaces/surfaceSecurity';

describe('generated Surface iframe policy', () => {
  test('permits React form events without granting the host origin', () => {
    const permissions = new Set(SURFACE_IFRAME_SANDBOX.split(/\s+/));

    expect(permissions).toContain('allow-scripts');
    expect(permissions).toContain('allow-forms');
    expect(permissions).not.toContain('allow-same-origin');
    expect(permissions).not.toContain('allow-popups');
    expect(permissions).not.toContain('allow-top-navigation');
  });
});
