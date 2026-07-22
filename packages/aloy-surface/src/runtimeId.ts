let fallbackSequence = 0;

type RuntimeIdDependencies = {
  crypto?: Pick<Crypto, 'getRandomValues' | 'randomUUID'> | null;
  now?: () => number;
  random?: () => number;
};

/**
 * Create a collision-resistant transport identifier inside an isolated Surface.
 *
 * Sandboxed opaque-origin frames do not consistently expose
 * `crypto.randomUUID()`. These identifiers are correlation and idempotency
 * values, not secrets, so a timestamp + monotonic sequence + random suffix is
 * a safe compatibility fallback. The host still owns authorization and input
 * validation for every request.
 */
export function createRuntimeId(
  prefix: string,
  dependencies: RuntimeIdDependencies = {},
): string {
  const runtimeCrypto = dependencies.crypto === undefined
    ? (typeof globalThis !== 'undefined' ? globalThis.crypto : undefined)
    : dependencies.crypto;
  if (typeof runtimeCrypto?.randomUUID === 'function') {
    return runtimeCrypto.randomUUID();
  }

  fallbackSequence = (fallbackSequence + 1) % Number.MAX_SAFE_INTEGER;
  const now = (dependencies.now ?? Date.now)().toString(36);
  let entropy: string;
  if (typeof runtimeCrypto?.getRandomValues === 'function') {
    const values = new Uint32Array(2);
    runtimeCrypto.getRandomValues(values);
    entropy = Array.from(values, (value) => value.toString(36)).join('');
  } else {
    const random = dependencies.random ?? Math.random;
    entropy = `${random().toString(36).slice(2)}${random().toString(36).slice(2)}`;
  }
  return `${prefix}-${now}-${fallbackSequence.toString(36)}-${entropy}`.slice(0, 200);
}
