import { useState } from 'react';
import {
  AlertTriangle,
  Bookmark,
  BookmarkCheck,
  Check,
  CheckCircle2,
  CircleSlash,
  Copy,
  Download,
  History,
  RotateCcw,
  ShieldCheck,
  XCircle,
} from 'lucide-react';
import { apiStreamFetch } from '@/api/client';
import { saveToLibrary } from '@/api/files';
import { FileTypeIcon } from '@/components/files/FileVisual';
import type { MessageResponse } from '@/types';
import { RunReplay } from './RunReplay';
import { Markdown } from './Markdown';

/** Auth'd download: the endpoint needs the Bearer header, so a plain <a href>
 *  can't do it — fetch the blob and click a transient object URL. */
async function downloadStoredFile(fileId: string, name: string) {
  try {
    const res = await apiStreamFetch(`/files/${fileId}`, undefined, undefined, 'GET');
    const url = URL.createObjectURL(await res.blob());
    const a = document.createElement('a');
    a.href = url;
    a.download = name;
    a.click();
    URL.revokeObjectURL(url);
  } catch {
    // transient; the chip stays usable for a retry
  }
}

export function MessageBubble({
  message,
  onOpenArtifact,
  onResend,
  onContinue,
}: {
  message: MessageResponse;
  onOpenArtifact?: (path: string) => void;
  onResend?: (content: string) => void;
  /** Continue an interrupted (stopped) response from where it left off. */
  onContinue?: (message: MessageResponse) => void;
}) {
  const isUser = message.role === 'user';
  const isSurfaceAction = message.metadata?.kind === 'surface_action_lifecycle';
  const surfaceActionStatus = message.metadata?.status ?? 'waiting_approval';
  const artifacts = message.metadata?.artifacts ?? [];
  const runId = message.metadata?.run_id ?? null;
  const [replaying, setReplaying] = useState(false);
  const [copied, setCopied] = useState(false);
  const [savedIds, setSavedIds] = useState<Set<string>>(new Set());

  async function handleSaveToLibrary(fileId: string) {
    try {
      await saveToLibrary(fileId);
      setSavedIds((prev) => new Set(prev).add(fileId));
    } catch {
      // leave the bookmark unfilled so the user can retry
    }
  }

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
      aria-label="Copy message"
      className={`flex h-7 w-7 items-center justify-center rounded-md text-zinc-500 transition hover:bg-zinc-800 hover:text-accent-600 ${
        copied ? 'opacity-100' : 'opacity-0 group-hover:opacity-100 group-focus-within:opacity-100'
      }`}
    >
      {copied ? <Check size={14} className="text-accent-600" /> : <Copy size={14} />}
    </button>
  );

  return (
    <article className={`group flex w-full ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`min-w-0 ${isUser ? 'max-w-[min(82%,42rem)] text-right' : 'w-full'}`}>
      <div
        className={`${
          isSurfaceAction
            ? 'max-w-3xl rounded-xl border border-zinc-700 bg-zinc-900/70 px-4 py-3 text-zinc-200 shadow-sm'
            : isUser
              ? 'rounded-2xl border border-accent-500/15 bg-accent-500/[0.06] px-4 py-3 text-[15px] leading-6 text-zinc-200 shadow-sm selection:bg-accent-500/20 sm:text-base'
              : 'text-[15px] leading-7 text-zinc-200 sm:text-base'
        }`}
      >
        {isSurfaceAction && (
          <div className="mb-2 flex items-center gap-2 border-b border-zinc-700/80 pb-2">
            {surfaceActionStatus === 'committed' ? (
              <CheckCircle2 size={15} className="text-emerald-500" />
            ) : surfaceActionStatus === 'indeterminate' ? (
              <AlertTriangle size={15} className="text-amber-500" />
            ) : ['failed', 'rejected'].includes(surfaceActionStatus) ? (
              <XCircle size={15} className="text-red-400" />
            ) : (
              <ShieldCheck size={15} className="text-accent-500" />
            )}
            <span className="text-xs font-semibold uppercase tracking-wide text-zinc-300">
              Surface action
            </span>
            <span className="ml-auto rounded-full bg-zinc-800 px-2 py-0.5 text-[11px] capitalize text-zinc-400">
              {surfaceActionStatus.replaceAll('_', ' ')}
            </span>
          </div>
        )}
        {(message.metadata?.images?.length ?? 0) > 0 && (
          <div className={`mb-3 flex flex-wrap gap-2 ${isUser ? 'justify-end' : ''}`}>
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
        {(message.metadata?.files?.length ?? 0) > 0 && (
          <div className={`mb-3 flex flex-wrap gap-2 ${isUser ? 'justify-end' : ''}`}>
            {message.metadata!.files!.map((f, i) => (
              <span
                key={i}
                className={`flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs ${
                  isUser
                    ? 'border-accent-500/15 bg-zinc-950/45 text-zinc-200'
                    : 'border-zinc-700 bg-zinc-900/60 text-zinc-200'
                }`}
              >
                <FileTypeIcon file={f} size={13} />
                <span className="max-w-44 truncate font-medium">{f.name}</span>
                <span className="opacity-60">
                  {f.size > 1024 * 1024
                    ? `${(f.size / (1024 * 1024)).toFixed(1)} MB`
                    : `${(f.size / 1024).toFixed(1)} KB`}
                </span>
                {f.file_id && (
                  <>
                    <button
                      type="button"
                      title="Download"
                      onClick={() => downloadStoredFile(f.file_id!, f.name)}
                      className="opacity-70 transition-opacity hover:opacity-100"
                    >
                      <Download size={12} />
                    </button>
                    <button
                      type="button"
                      title={
                        savedIds.has(f.file_id)
                          ? 'Saved to My Files'
                          : 'Save to My Files (remembered across all chats)'
                      }
                      onClick={() => handleSaveToLibrary(f.file_id!)}
                      className="opacity-70 transition-opacity hover:opacity-100"
                    >
                      {savedIds.has(f.file_id) ? (
                        <BookmarkCheck size={12} />
                      ) : (
                        <Bookmark size={12} />
                      )}
                    </button>
                  </>
                )}
              </span>
            ))}
          </div>
        )}
        {isUser ? (
          <p className="whitespace-pre-wrap break-words">{message.content}</p>
        ) : (
          <div>
            <Markdown>{message.content}</Markdown>
          </div>
        )}

        {!isUser && message.metadata?.stopped && (
          <div className="mt-2 flex items-center gap-1.5 rounded-lg border border-zinc-700 bg-zinc-900/60 px-2.5 py-1.5 text-xs text-zinc-400">
            <CircleSlash size={12} className="shrink-0" />
            <span className="flex-1">Response was interrupted.</span>
            {onContinue && (
              <button
                type="button"
                onClick={() => onContinue(message)}
                className="rounded-md bg-zinc-700 px-2 py-0.5 font-medium text-zinc-200 transition-colors hover:bg-accent-600 hover:text-white"
              >
                Continue
              </button>
            )}
          </div>
        )}

        {!isUser && artifacts.length > 0 && (
          <div className="mt-4 space-y-1 border-t border-zinc-800 pt-3">
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
                  <FileTypeIcon file={{ name: path ?? String(a.tool_name ?? 'artifact'), kind: 'artifact' }} size={12} />
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
          <div className="mt-4 flex items-center gap-3 border-t border-zinc-800 pt-3 text-xs text-zinc-400">
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

        <div className={`mt-2 flex min-h-7 items-center gap-1 ${isUser ? 'justify-end' : 'justify-start'}`}>
          {isUser && onResend && (
            <button
              type="button"
              onClick={() => onResend(message.content)}
              title="Send again"
              aria-label="Send again"
              className="flex h-7 w-7 items-center justify-center rounded-md text-zinc-500 opacity-0 transition hover:bg-zinc-800 hover:text-accent-600 group-hover:opacity-100 group-focus-within:opacity-100"
            >
              <RotateCcw size={14} />
            </button>
          )}
          {copyButton}
        </div>
      </div>

      {replaying && runId && (
        <RunReplay runId={runId} onClose={() => setReplaying(false)} />
      )}
    </article>
  );
}
