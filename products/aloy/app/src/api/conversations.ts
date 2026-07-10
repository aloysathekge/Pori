import { apiFetch, apiUploadFile } from './client';
import type {
  ConversationResponse,
  ConversationDetail,
  MessageResponse,
} from '@/types';

export function listConversations(limit = 20, offset = 0) {
  return apiFetch<ConversationResponse[]>(
    `/conversations?limit=${limit}&offset=${offset}`,
  );
}

export function getConversation(id: string) {
  return apiFetch<ConversationDetail>(`/conversations/${id}`);
}

export function createConversation(data: {
  title?: string | null;
  agent_config_id?: string | null;
}) {
  return apiFetch<ConversationResponse>('/conversations', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export function updateConversation(id: string, title: string) {
  return apiFetch<ConversationResponse>(`/conversations/${id}`, {
    method: 'PATCH',
    body: JSON.stringify({ title }),
  });
}

export function deleteConversation(id: string) {
  return apiFetch<void>(`/conversations/${id}`, { method: 'DELETE' });
}

export interface UploadedFileRef {
  file_id: string;
  name: string;
  size_bytes: number;
  content_type: string;
}

/** Durable upload (rung 4): the file goes to object storage + the sandbox;
 *  the message later carries only its file_id reference. */
export function uploadConversationFile(
  conversationId: string,
  file: File,
  onProgress?: (pct: number) => void,
) {
  return apiUploadFile<UploadedFileRef>(
    `/conversations/${conversationId}/files`,
    file,
    onProgress,
  );
}

/** Stop the conversation's in-flight run (agent halts at the next step). */
export function stopGeneration(id: string) {
  return apiFetch<{ status: string }>(`/conversations/${id}/stop`, {
    method: 'POST',
  });
}

export function sendMessage(
  conversationId: string,
  data: { content: string; max_steps?: number; stream?: boolean; team_id?: string | null },
) {
  return apiFetch<MessageResponse>(
    `/conversations/${conversationId}/messages`,
    { method: 'POST', body: JSON.stringify(data) },
  );
}
