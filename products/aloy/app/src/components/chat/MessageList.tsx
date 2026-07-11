import { useCallback, useEffect, useRef } from 'react';
import { MessageBubble } from '@/components/chat/MessageBubble';
import { StreamingIndicator } from '@/components/chat/StreamingIndicator';
import { ClarifyPrompt } from '@/components/chat/ClarifyPrompt';
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
  onOpenArtifact: (path: string) => void;
  /** Omit while sending — the bubbles hide their resend/continue actions. */
  onResend?: (content: string) => void;
  onContinue?: (message: MessageResponse) => void;
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
  onOpenArtifact,
  onResend,
  onContinue,
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
      <div ref={messagesEndRef} />
    </div>
  );
}
