import { useCallback, useState } from 'react';
import {
  createConversation as apiCreateConversation,
  deleteConversation as apiDeleteConversation,
  listConversations,
} from '@/api/conversations';
import type { ConversationResponse } from '@/types';

/**
 * Owns the conversation list + active selection. Page-agnostic by design
 * (Surfaces will reuse it): routing (navigate-on-create/delete,
 * redirect-to-most-recent) stays with the caller, which reacts to the
 * returned values instead.
 */
export function useConversations() {
  const [conversations, setConversations] = useState<ConversationResponse[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);

  const loadConversations = useCallback(async () => {
    try {
      const convos = await listConversations(50);
      setConversations(convos);
    } catch {
      // handle silently
    }
  }, []);

  /** Create a conversation and prepend it to the list. Returns the new
   *  conversation so the caller can navigate to it (null on failure). */
  const createConversation =
    useCallback(async (): Promise<ConversationResponse | null> => {
      try {
        const convo = await apiCreateConversation({});
        setConversations((prev) => [convo, ...prev]);
        return convo;
      } catch {
        // handle silently
        return null;
      }
    }, []);

  /** Delete a conversation and drop it from the list. Returns true on
   *  success so the caller can navigate away if it was the active one. */
  const deleteConversation = useCallback(async (id: string): Promise<boolean> => {
    try {
      await apiDeleteConversation(id);
      setConversations((prev) => prev.filter((c) => c.id !== id));
      return true;
    } catch {
      // handle silently
      return false;
    }
  }, []);

  return {
    conversations,
    activeId,
    setActiveId,
    loadConversations,
    createConversation,
    deleteConversation,
  };
}
