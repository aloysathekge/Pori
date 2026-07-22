import { describe, expect, test } from 'bun:test';
import {
  buildHtmlArtifactPreviewDocument,
  HTML_ARTIFACT_CSP,
  HTML_ARTIFACT_IFRAME_SANDBOX,
} from '../src/components/workbench/htmlArtifactSecurity';

describe('HTML artifact preview security', () => {
  test('keeps the iframe opaque and without host or download authority', () => {
    expect(HTML_ARTIFACT_IFRAME_SANDBOX.split(' ').sort()).toEqual([
      'allow-forms',
      'allow-scripts',
    ]);
    expect(HTML_ARTIFACT_IFRAME_SANDBOX).not.toContain('allow-same-origin');
    expect(HTML_ARTIFACT_IFRAME_SANDBOX).not.toContain('allow-downloads');
    expect(HTML_ARTIFACT_IFRAME_SANDBOX).not.toContain('allow-top-navigation');
  });

  test('injects a fail-closed CSP and local navigation guards before artifact code', () => {
    const source = '<html><head><title>Plan</title></head><body><script>window.ready=true</script></body></html>';
    const document = buildHtmlArtifactPreviewDocument(source);

    expect(HTML_ARTIFACT_CSP).toContain("default-src 'none'");
    expect(HTML_ARTIFACT_CSP).toContain("connect-src 'none'");
    expect(HTML_ARTIFACT_CSP).toContain("form-action 'none'");
    expect(HTML_ARTIFACT_CSP).toContain("navigate-to 'none'");
    expect(document.indexOf('Content-Security-Policy')).toBeLessThan(
      document.indexOf('window.ready=true'),
    );
    expect(document).toContain("addEventListener('submit',preventNavigation,false)");
    expect(document).not.toContain('stopImmediatePropagation');
    expect(document).toContain(source.slice(source.indexOf('<body>')));
  });
});
