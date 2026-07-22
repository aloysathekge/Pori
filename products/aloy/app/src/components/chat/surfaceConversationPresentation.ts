import type { MessageResponse } from '@/types';

export interface SurfaceConversationPresentation {
  card: boolean;
  content: string;
  status: 'sent' | 'failed' | 'stopped' | null;
}

export function surfaceConversationPresentation(
  message: MessageResponse,
): SurfaceConversationPresentation {
  const kind = message.metadata?.kind;
  const label = message.metadata?.surface_request_label?.trim();
  if (message.role === 'user' && kind === 'surface_interaction') {
    return {
      card: true,
      content: label || 'Asked Aloy to continue from the Surface.',
      status: 'sent',
    };
  }
  if (kind === 'surface_reasoning_result' && message.metadata?.status === 'failed') {
    return {
      card: true,
      content: label
        ? `Aloy couldn't complete ${label} right now. Try again from the Surface.`
        : "Aloy couldn't complete this Surface request right now. Try again from the Surface.",
      status: 'failed',
    };
  }
  if (kind === 'surface_reasoning_result' && message.metadata?.status === 'cancelled') {
    return {
      card: true,
      content: label
        ? `${label} was stopped. You can restart it from the Surface.`
        : 'This Surface request was stopped. You can restart it from the Surface.',
      status: 'stopped',
    };
  }
  return { card: false, content: message.content, status: null };
}

/**
 * A generated control is not a second user turn. Keep its request card while
 * work is active, then let the durable outcome replace it in conversation.
 * Direct questions typed by the user inside a Surface remain visible.
 */
export function collapseResolvedSurfaceRequests(
  messages: MessageResponse[],
): MessageResponse[] {
  const outcomeIds = new Set(
    messages
      .filter((message) => message.metadata?.kind === 'surface_reasoning_result')
      .map((message) => message.metadata?.surface_interaction_id)
      .filter((value): value is string => Boolean(value)),
  );
  return messages.filter((message) => {
    if (
      message.role !== 'user'
      || message.metadata?.kind !== 'surface_interaction'
      || !message.metadata.surface_interaction_id
      || !outcomeIds.has(message.metadata.surface_interaction_id)
    ) return true;
    return message.metadata.surface_request_origin === 'user_question'
      || (
        message.metadata.surface_request_origin == null
        && !message.content.startsWith('Carry out the ')
      );
  });
}
