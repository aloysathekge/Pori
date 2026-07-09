import { apiFetch } from './client';

export interface ArtifactInfo {
  path: string;
  tool_name: string | null;
  bytes_written: number | null;
  message_id: string;
}

export interface ArtifactContent {
  path: string;
  content: string;
  language: string;
  truncated: boolean;
}

export function listArtifacts(conversationId: string) {
  return apiFetch<ArtifactInfo[]>(`/conversations/${conversationId}/artifacts`);
}

export function getArtifactContent(conversationId: string, path: string) {
  return apiFetch<ArtifactContent>(
    `/conversations/${conversationId}/artifacts/content?path=${encodeURIComponent(path)}`,
  );
}
