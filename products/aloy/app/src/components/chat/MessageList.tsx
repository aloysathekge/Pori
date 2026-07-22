import { useCallback, useEffect, useMemo, useRef, type ReactNode } from 'react';
import { MessageBubble } from '@/components/chat/MessageBubble';
import { WorkStory } from '@/components/chat/WorkStory';
import { ClarifyPrompt } from '@/components/chat/ClarifyPrompt';
import { ApprovalPrompt } from '@/components/chat/ApprovalPrompt';
import type { ApprovalDecision } from '@/api/sse';
import type { MessageResponse, RunTimelineEvent } from '@/types';
import { collapseResolvedSurfaceRequests } from './surfaceConversationPresentation';

function conversationTimeLabel(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '';
  const today = new Date();
  const yesterday = new Date(today);
  yesterday.setDate(today.getDate() - 1);
  const sameDay = (left: Date, right: Date) =>
    left.getFullYear() === right.getFullYear()
    && left.getMonth() === right.getMonth()
    && left.getDate() === right.getDate();
  const day = sameDay(date, today)
    ? 'Today'
    : sameDay(date, yesterday)
      ? 'Yesterday'
      : new Intl.DateTimeFormat(undefined, { month: 'short', day: 'numeric' }).format(date);
  const time = new Intl.DateTimeFormat(undefined, { hour: 'numeric', minute: '2-digit' }).format(date);
  return `${day} ${time}`;
}

interface Props {
  messages: MessageResponse[];
  streaming: boolean;
  streamText: string;
  streamStory: RunTimelineEvent[];
  clarify: { question: string; options: string[] } | null;
  onAnswerClarify: (value: string) => void;
  approval: { tool: string; arguments: Record<string, unknown> } | null;
  onDecideApproval: (decision: ApprovalDecision) => void;
  onOpenArtifact: (path: string) => void;
  /** Omit while sending — the bubbles hide their resend/continue actions. */
  onResend?: (content: string) => void;
  onContinue?: (message: MessageResponse) => void;
  hasOlder?: boolean;
  loadingOlder?: boolean;
  onLoadOlder?: () => void;
  afterMessages?: ReactNode;
}

/**
 * The message column: history bubbles, the live streaming bubble + activity
 * indicator, the inline clarify panel, and the auto-scroll anchor.
 */
export function MessageList({
  messages,
  streaming,
  streamText,
  streamStory,
  clarify,
  onAnswerClarify,
  approval,
  onDecideApproval,
  onOpenArtifact,
  onResend,
  onContinue,
  hasOlder,
  loadingOlder,
  onLoadOlder,
  afterMessages,
}: Props) {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const visibleMessages = useMemo(
    () => collapseResolvedSurfaceRequests(messages),
    [messages],
  );

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, streaming, scrollToBottom]);

  return (
    <div className="mx-auto max-w-[56rem] space-y-10 pb-2">
      {hasOlder && onLoadOlder && (
        <div className="flex justify-center">
          <button
            type="button"
            onClick={onLoadOlder}
            disabled={loadingOlder}
            className="rounded-full border border-zinc-800 px-3 py-1.5 text-xs font-medium text-zinc-400 transition hover:border-zinc-700 hover:text-zinc-200 disabled:opacity-60"
          >
            {loadingOlder ? 'Loading earlier messages…' : 'Load earlier messages'}
          </button>
        </div>
      )}
      {visibleMessages.map((msg, index) => {
        const previous = visibleMessages[index - 1];
        const date = new Date(msg.created_at);
        const previousDate = previous ? new Date(previous.created_at) : null;
        const startsDay = !previousDate
          || date.getFullYear() !== previousDate.getFullYear()
          || date.getMonth() !== previousDate.getMonth()
          || date.getDate() !== previousDate.getDate();
        return (
          <div key={msg.id} className={startsDay ? 'space-y-7' : undefined}>
            {startsDay && (
              <p className="text-center text-xs font-medium text-zinc-500">
                {conversationTimeLabel(msg.created_at)}
              </p>
            )}
            <MessageBubble
              message={msg}
              onOpenArtifact={onOpenArtifact}
              onResend={onResend}
              onContinue={onContinue}
            />
          </div>
        );
      })}
      {streaming && (
        <div>
          <WorkStory entries={streamStory} live />
          {streamText && (
            <MessageBubble
              message={{
                id: 'streaming',
                role: 'assistant',
                content: streamText,
                metadata: null,
                created_at: new Date().toISOString(),
              }}
            />
          )}
        </div>
      )}
      {clarify && (
        <ClarifyPrompt
          question={clarify.question}
          options={clarify.options}
          onAnswer={onAnswerClarify}
        />
      )}
      {approval && (
        <ApprovalPrompt
          tool={approval.tool}
          args={approval.arguments}
          onDecide={onDecideApproval}
        />
      )}
      {afterMessages}
      <div ref={messagesEndRef} />
    </div>
  );
}
