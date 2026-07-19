import { useCallback, useRef, useState } from 'react';
import { listConversationFiles } from '@/api/files';
import type { StoredFileReference } from './useAttachments';

export function useFileReferences(conversationId: string | null) {
  const [state, setState] = useState<{
    conversationId: string | null;
    files: StoredFileReference[];
    loading: boolean;
    error: string;
  }>({ conversationId: null, files: [], loading: false, error: '' });
  const requestId = useRef(0);
  const current = state.conversationId === conversationId
    ? state
    : { conversationId, files: [], loading: false, error: '' };

  const search = useCallback(async (query: string) => {
    if (!conversationId) return;
    const currentRequest = ++requestId.current;
    setState((previous) => ({
      conversationId,
      files: previous.conversationId === conversationId ? previous.files : [],
      loading: true,
      error: '',
    }));
    try {
      const rows = await listConversationFiles(conversationId, query);
      if (currentRequest !== requestId.current) return;
      setState({
        conversationId,
        loading: false,
        error: '',
        files: rows.map((row) => ({
          file_id: row.file_id,
          name: row.name,
          size: row.size_bytes,
          content_type: row.content_type,
          kind: row.kind,
          event_id: row.event_id,
          event_title: row.event_title,
          created_at: row.created_at,
        })),
      });
    } catch (cause) {
      if (currentRequest !== requestId.current) return;
      setState({
        conversationId,
        files: [],
        loading: false,
        error: cause instanceof Error ? cause.message : 'Could not load files',
      });
    }
  }, [conversationId]);

  return { files: current.files, loading: current.loading, error: current.error, search };
}
