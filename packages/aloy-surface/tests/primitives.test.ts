import { describe, expect, test } from 'bun:test';
import {
  SurfaceGrid,
  SurfaceRoot,
  SurfaceStack,
  surfaceTokens,
} from '../src/index';

describe('@aloy/surface primitives', () => {
  test('exports stable responsive tokens', () => {
    expect(surfaceTokens.space.lg).toBe('1rem');
    expect(surfaceTokens.breakpoint.mobile).toBe('30rem');
    expect(surfaceTokens.color.accent).toMatch(/^#/);
  });

  test('root and layout primitives carry host-neutral identity attributes', () => {
    const root = SurfaceRoot({ children: 'content' });
    const stack = SurfaceStack({ children: 'content' });
    const grid = SurfaceGrid({ children: 'content' });

    expect(root.props['data-aloy-surface-root']).toBe('true');
    expect(stack.props.className).toContain('aloy-surface-stack');
    expect(stack.props.style.minWidth).toBe(0);
    expect(grid.props.className).toContain('aloy-surface-grid');
    expect(grid.props.style.gridTemplateColumns).toContain('auto-fit');
  });
});
