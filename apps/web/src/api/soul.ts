import { apiFetch } from './client';
import type { SoulResponse } from '@/types';

/** The user's standalone SOUL identity document (the SOUL.md equivalent). */
export function getSoul() {
  return apiFetch<SoulResponse>('/me/soul');
}

export function setSoul(content: string) {
  return apiFetch<SoulResponse>('/me/soul', {
    method: 'PUT',
    body: JSON.stringify({ content }),
  });
}

export function clearSoul() {
  return apiFetch<void>('/me/soul', { method: 'DELETE' });
}
