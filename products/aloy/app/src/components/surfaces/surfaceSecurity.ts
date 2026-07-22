/**
 * Generated Surfaces may run React form event handlers, but they never share
 * the host origin. The host-owned runtime CSP separately keeps form-action at
 * `none`, so forms can emit typed bridge commands without navigating or
 * transmitting data themselves.
 */
export const SURFACE_IFRAME_SANDBOX = 'allow-scripts allow-forms';
