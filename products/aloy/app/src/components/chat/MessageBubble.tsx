import { Bot, FileText, User } from 'lucide-react';
import type { MessageResponse } from '@/types';

export function MessageBubble({ message }: { message: MessageResponse }) {
  const isUser = message.role === 'user';
  const artifacts = message.metadata?.artifacts ?? [];

  return (
    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : ''}`}>
      <div
        className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full ${
          isUser ? 'bg-accent-600' : 'bg-zinc-700'
        }`}
      >
        {isUser ? <User size={16} /> : <Bot size={16} />}
      </div>
      <div
        className={`max-w-2xl rounded-2xl px-4 py-3 text-sm leading-relaxed ${
          isUser
            ? 'bg-accent-600 text-white'
            : 'bg-zinc-800 text-zinc-200'
        }`}
      >
        <p className="whitespace-pre-wrap">{message.content}</p>

        {!isUser && artifacts.length > 0 && (
          <div className="mt-2 space-y-1 border-t border-zinc-700 pt-2">
            <p className="text-xs font-medium text-zinc-400">Files</p>
            {artifacts.map((a, i) => (
              <div
                key={`${a.path ?? 'artifact'}-${i}`}
                className="flex items-center gap-1.5 text-xs text-zinc-400"
              >
                <FileText size={12} className="shrink-0 text-zinc-500" />
                <span className="truncate font-mono">
                  {a.path ?? a.tool_name ?? 'artifact'}
                </span>
                {typeof a.bytes_written === 'number' && (
                  <span className="shrink-0 text-zinc-600">
                    ({a.bytes_written} B)
                  </span>
                )}
              </div>
            ))}
          </div>
        )}

        {message.metadata?.steps_taken != null && !isUser && (
          <div className="mt-2 flex items-center gap-2 border-t border-zinc-700 pt-2 text-xs text-zinc-400">
            <span>{message.metadata.steps_taken} steps</span>
            {message.metadata.reasoning && (
              <details className="ml-2">
                <summary className="cursor-pointer hover:text-zinc-300">
                  Reasoning
                </summary>
                <p className="mt-1 text-zinc-400">
                  {message.metadata.reasoning}
                </p>
              </details>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
