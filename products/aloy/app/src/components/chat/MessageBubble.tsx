import { useState } from 'react';
import { Check, Copy, FileText, History } from 'lucide-react';
import type { MessageResponse } from '@/types';
import { RunReplay } from './RunReplay';
import { Markdown } from './Markdown';

export function MessageBubble({
  message,
  onOpenArtifact,
}: {
  message: MessageResponse;
  onOpenArtifact?: (path: string) => void;
}) {
  const isUser = message.role === 'user';
  const artifacts = message.metadata?.artifacts ?? [];
  const runId = message.metadata?.run_id ?? null;
  const [replaying, setReplaying] = useState(false);
  const [copied, setCopied] = useState(false);

  function copyMessage() {
    navigator.clipboard.writeText(message.content).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }

  const copyButton = (
    <button
      type="button"
      onClick={copyMessage}
      title="Copy message"
      className={`self-end pb-1 text-zinc-500 transition-opacity hover:text-accent-600 ${
        copied ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'
      }`}
    >
      {copied ? <Check size={14} className="text-accent-600" /> : <Copy size={14} />}
    </button>
  );

  return (
    <div className={`group flex gap-2 ${isUser ? 'justify-end' : 'justify-start'}`}>
      {isUser && copyButton}
      <div
        className={`max-w-3xl rounded-2xl px-4 py-3 text-sm leading-relaxed ${
          isUser
            ? 'bg-accent-600 text-white selection:bg-white/35 selection:text-white'
            : 'bg-zinc-800 text-zinc-200'
        }`}
      >
        {(message.metadata?.images?.length ?? 0) > 0 && (
          <div className="mb-2 flex flex-wrap gap-2">
            {message.metadata!.images!.map((img, i) => (
              <img
                key={i}
                src={`data:${img.media_type};base64,${img.data}`}
                alt={`attachment ${i + 1}`}
                className="max-h-64 max-w-full rounded-lg border border-zinc-700 object-contain"
              />
            ))}
          </div>
        )}
        {isUser ? (
          <p className="whitespace-pre-wrap break-words">{message.content}</p>
        ) : (
          <div className="text-sm">
            <Markdown>{message.content}</Markdown>
          </div>
        )}

        {!isUser && artifacts.length > 0 && (
          <div className="mt-2 space-y-1 border-t border-zinc-700 pt-2">
            <p className="text-xs font-medium text-zinc-400">Files</p>
            {artifacts.map((a, i) => {
              const path = a.path as string | undefined;
              const openable = Boolean(path && onOpenArtifact);
              return (
                <button
                  key={`${path ?? 'artifact'}-${i}`}
                  type="button"
                  disabled={!openable}
                  onClick={() => path && onOpenArtifact?.(path)}
                  className={`flex w-full items-center gap-1.5 rounded px-1 py-0.5 text-left text-xs text-zinc-400 ${
                    openable ? 'hover:bg-zinc-700/60 hover:text-accent-500' : ''
                  }`}
                >
                  <FileText size={12} className="shrink-0 text-zinc-500" />
                  <span className="truncate font-mono">
                    {path ?? a.tool_name ?? 'artifact'}
                  </span>
                  {typeof a.bytes_written === 'number' && (
                    <span className="shrink-0 text-zinc-600">
                      ({a.bytes_written} B)
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        )}

        {message.metadata?.steps_taken != null && !isUser && (
          <div className="mt-2 flex items-center gap-3 border-t border-zinc-700 pt-2 text-xs text-zinc-400">
            <span>{message.metadata.steps_taken} steps</span>
            {message.metadata.reasoning && (
              <details className="ml-1">
                <summary className="cursor-pointer hover:text-zinc-300">
                  Reasoning
                </summary>
                <p className="mt-1 text-zinc-400">
                  {message.metadata.reasoning}
                </p>
              </details>
            )}
            {runId && (
              <button
                type="button"
                onClick={() => setReplaying(true)}
                className="ml-auto inline-flex items-center gap-1 text-zinc-500 hover:text-accent-600"
              >
                <History size={12} /> Replay
              </button>
            )}
          </div>
        )}
      </div>
      {!isUser && copyButton}

      {replaying && runId && (
        <RunReplay runId={runId} onClose={() => setReplaying(false)} />
      )}
    </div>
  );
}
