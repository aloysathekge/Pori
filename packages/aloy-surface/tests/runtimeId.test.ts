import { describe, expect, test } from 'bun:test';

import { createRuntimeId } from '../src/runtimeId';

describe('isolated Surface runtime identifiers', () => {
  test('uses native randomUUID when the runtime provides it', () => {
    expect(createRuntimeId('request', {
      crypto: {
        randomUUID: () => 'native-id' as `${string}-${string}-${string}-${string}-${string}`,
        getRandomValues: (values) => values,
      },
    })).toBe('native-id');
  });

  test('stays unique without secure-context crypto APIs', () => {
    const dependencies = {
      crypto: null,
      now: () => 1_700_000_000_000,
      random: () => 0.25,
    };

    const first = createRuntimeId('command', dependencies);
    const second = createRuntimeId('command', dependencies);

    expect(first).toStartWith('command-');
    expect(second).toStartWith('command-');
    expect(first).not.toBe(second);
    expect(first.length).toBeLessThanOrEqual(200);
  });

  test('uses getRandomValues when randomUUID is unavailable', () => {
    const id = createRuntimeId('request', {
      crypto: {
        randomUUID: undefined,
        getRandomValues: (values) => {
          values[0] = 123;
          values[1] = 456;
          return values;
        },
      },
      now: () => 42,
    });

    expect(id).toContain('-3fco');
  });
});
