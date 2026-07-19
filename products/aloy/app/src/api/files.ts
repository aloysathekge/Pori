import { apiBlobFetch, apiFetch, apiUploadFile } from './client';

/** A durable stored file (object storage pointer). */
export interface StoredFileView {
  file_id: string;
  name: string;
  size_bytes: number;
  content_type: string;
  kind: 'upload' | 'artifact';
  in_library: boolean;
  event_id: string;
  event_title?: string;
  conversation_id: string | null;
  created_at: string;
}

/** The caller's retained files. Runtime access remains Event-scoped. */
export function listMyFiles() {
  return apiFetch<StoredFileView[]>('/files');
}

export function uploadLibraryFile(
  file: File,
  onProgress?: (pct: number) => void,
) {
  return apiUploadFile<StoredFileView>('/files', file, onProgress);
}

export function listConversationFiles(conversationId: string, query = '') {
  const params = new URLSearchParams({ limit: '50' });
  if (query.trim()) params.set('q', query.trim());
  return apiFetch<StoredFileView[]>(
    `/conversations/${conversationId}/files?${params.toString()}`,
  );
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
