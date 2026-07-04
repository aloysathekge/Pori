import { apiFetch } from './client';
import type {
  MemoryBlock,
  KnowledgeEntry,
  KnowledgeEntryCreate,
  KnowledgeSearchRequest,
} from '@/types';

export function getMemory() {
  return apiFetch<{ blocks: MemoryBlock[] }>('/me/memory');
}

export function getMemoryBlock(label: string) {
  return apiFetch<MemoryBlock>(`/me/memory/${label}`);
}

export function updateMemoryBlock(label: string, value: string) {
  return apiFetch<MemoryBlock>(`/me/memory/${label}`, {
    method: 'PATCH',
    body: JSON.stringify({ value }),
  });
}

export function resetMemory() {
  return apiFetch<void>('/me/memory', { method: 'DELETE' });
}

export function listKnowledgeEntries(limit = 50, offset = 0) {
  return apiFetch<KnowledgeEntry[]>(
    `/me/memory/knowledge?limit=${limit}&offset=${offset}`,
  );
}

export function createKnowledgeEntry(data: KnowledgeEntryCreate) {
  return apiFetch<KnowledgeEntry>('/me/memory/knowledge', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export function searchKnowledgeEntries(body: KnowledgeSearchRequest) {
  return apiFetch<KnowledgeEntry[]>('/me/memory/archival/search', {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

export function deleteKnowledgeEntry(id: string) {
  return apiFetch<void>(`/me/memory/knowledge/${id}`, { method: 'DELETE' });
}

/** Backward-compatible aliases over knowledge endpoints */
export function listArchivalMemory(limit = 50, offset = 0) {
  return listKnowledgeEntries(limit, offset);
}

export function searchArchivalMemory(
  query: string,
  k = 10,
  tags?: string[],
) {
  return apiFetch<KnowledgeEntry[]>('/me/memory/archival/search', {
    method: 'POST',
    body: JSON.stringify({ query, k, tags: tags ?? null }),
  });
}

export function deleteArchivalPassage(id: string) {
  return deleteKnowledgeEntry(id);
}
