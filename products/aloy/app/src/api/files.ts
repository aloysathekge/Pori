import { apiBlobFetch, apiFetch } from './client';

/** A durable stored file (object storage pointer). */
export interface StoredFileView {
  file_id: string;
  name: string;
  size_bytes: number;
  content_type: string;
  kind: 'upload' | 'artifact';
  in_library: boolean;
  conversation_id: string;
  created_at: string;
}

/** The caller's file library — files the agent always knows exist. */
export function listMyFiles() {
  return apiFetch<StoredFileView[]>('/files');
}

/** Save to the library: writes the memory pointer so any future chat can
 *  say "use my CV" and the agent knows what that means. */
export function saveToLibrary(fileId: string) {
  return apiFetch<StoredFileView>(`/files/${fileId}/library`, { method: 'POST' });
}

export function removeFromLibrary(fileId: string) {
  return apiFetch<StoredFileView>(`/files/${fileId}/library`, { method: 'DELETE' });
}

export function getStoredFileBlob(fileId: string) {
  return apiBlobFetch(`/files/${fileId}`);
}
