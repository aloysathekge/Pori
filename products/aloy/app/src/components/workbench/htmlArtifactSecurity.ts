/**
 * HTML artifacts may execute only inside an opaque-origin iframe. They have no
 * Surface bridge, no host origin, no network, no storage, and no permission to
 * navigate or download. `allow-forms` lets the artifact run normal form event
 * handlers; the CSP and bubble-phase guard still prevent browser submission.
 */
export const HTML_ARTIFACT_IFRAME_SANDBOX = 'allow-scripts allow-forms';

export const HTML_ARTIFACT_CSP = [
  "default-src 'none'",
  "base-uri 'none'",
  "connect-src 'none'",
  "font-src data:",
  "form-action 'none'",
  "frame-src 'none'",
  "img-src data: blob:",
  "media-src data: blob:",
  "navigate-to 'none'",
  "object-src 'none'",
  "script-src 'unsafe-inline'",
  "style-src 'unsafe-inline'",
  "worker-src 'none'",
].join('; ');

const PREVIEW_GUARD = `<script>(()=>{
  const preventNavigation=(event)=>{event.preventDefault();};
  addEventListener('submit',preventNavigation,false);
  addEventListener('auxclick',(event)=>{if(event.target?.closest?.('a'))preventNavigation(event);},false);
  addEventListener('click',(event)=>{if(event.target?.closest?.('a'))preventNavigation(event);},false);
  try{Object.defineProperty(window,'open',{value:()=>null,writable:false,configurable:false});}catch{}
  try{Object.defineProperty(navigator,'sendBeacon',{value:()=>false,writable:false,configurable:false});}catch{}
})();</script>`;

export function buildHtmlArtifactPreviewDocument(source: string) {
  const policy = `<meta http-equiv="Content-Security-Policy" content="${HTML_ARTIFACT_CSP}">`;
  const viewport = '<meta name="viewport" content="width=device-width,initial-scale=1">';
  const boundary = `${policy}${viewport}${PREVIEW_GUARD}`;
  // The trusted boundary is always serialized before every byte of artifact
  // content. Browsers tolerate a nested document shell and still apply its
  // styles/body content, while no source script can appear ahead of the CSP.
  return `<!doctype html><html><head>${boundary}</head><body>${source}</body></html>`;
}
