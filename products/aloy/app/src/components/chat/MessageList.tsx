import { useCallback, useEffect, useRef, type ReactNode } from 'react';
import { MessageBubble } from '@/components/chat/MessageBubble';
import { StreamingIndicator } from '@/components/chat/StreamingIndicator';
import { ClarifyPrompt } from '@/components/chat/ClarifyPrompt';
import { ApprovalPrompt } from '@/components/chat/ApprovalPrompt';
import type { ApprovalDecision } from '@/api/sse';
import type { MessageResponse, PlanItem, SSEToolEvent } from '@/types';

interface Props {
  messages: MessageResponse[];
  streaming: boolean;
  streamText: string;
  streamStatus: string;
  streamActivity: string;
  streamPlan: PlanItem[];
  streamTools: SSEToolEvent[];
  streamStep?: { step: number; max_steps: number };
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
  streamStatus,
  streamActivity,
  streamPlan,
  streamTools,
  streamStep,
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

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, streaming, scrollToBottom]);

  return (
    <div className="mx-auto max-w-4xl space-y-6">
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
      {messages.map((msg) => (
        <MessageBubble
          key={msg.id}
          message={msg}
          onOpenArtifact={onOpenArtifact}
          onResend={onResend}
          onContinue={onContinue}
        />
      ))}
      {streaming && streamText && (
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
      {streaming && (
        <StreamingIndicator
          status={streamStatus}
          activity={streamActivity}
          plan={streamPlan}
          tools={streamTools}
          step={streamStep}
        />
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
