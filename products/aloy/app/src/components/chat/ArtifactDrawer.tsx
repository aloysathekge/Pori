import { useCallback, useEffect, useState } from 'react';
import { Check, Copy, FileCode, FileText, X } from 'lucide-react';
import { Spinner } from '@/components/ui/Spinner';
import {
  getArtifactContent,
  listArtifacts,
  type ArtifactContent,
  type ArtifactInfo,
} from '@/api/artifacts';
import { Markdown } from './Markdown';

/** Slide-over panel showing files the agent wrote — code rendered as mono, .md
 *  rendered as markdown — with a switcher for every artifact in the conversation. */
export function ArtifactDrawer({
  conversationId,
  openPath,
  onClose,
}: {
  conversationId: string;
  openPath: string;
  onClose: () => void;
}) {
  const [files, setFiles] = useState<ArtifactInfo[]>([]);
  const [active, setActive] = useState(openPath);

  // useState only reads openPath on first mount; the drawer stays mounted, so
  // clicking a different artifact must explicitly switch the active file.
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setActive(openPath);
  }, [openPath]);
  const [content, setContent] = useState<ArtifactContent | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    listArtifacts(conversationId)
      .then(setFiles)
      .catch(() => {});
  }, [conversationId]);

  const load = useCallback(
    async (path: string) => {
      setLoading(true);
      setError('');
      try {
        setContent(await getArtifactContent(conversationId, path));
      } catch (e) {
        setContent(null);
        setError(e instanceof Error ? e.message : 'Could not open file');
      } finally {
        setLoading(false);
      }
    },
    [conversationId],
  );

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    load(active);
  }, [active, load]);

  function copy() {
    if (!content) return;
    navigator.clipboard.writeText(content.content).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }

  const isMd = content?.language === 'markdown';
  const name = (p: string) => p.split(/[/\\]/).pop() || p;

  return (
    <div className="fixed inset-0 z-40 flex justify-end">
      <div className="absolute inset-0 bg-black/40" onClick={onClose} />
      <div className="relative flex h-full w-full max-w-2xl flex-col border-l border-zinc-800 bg-zinc-900 shadow-xl">
        <div className="flex items-center justify-between border-b border-zinc-800 px-4 py-3">
          <div className="flex items-center gap-2 truncate">
            <FileCode size={16} className="shrink-0 text-accent-600" />
            <span className="truncate font-mono text-sm text-zinc-100">
              {name(active)}
            </span>
            {content?.truncated && (
              <span className="shrink-0 rounded bg-amber-50 px-1.5 py-0.5 text-xs text-amber-700">
                truncated
              </span>
            )}
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={copy}
              disabled={!content}
              className="rounded p-1.5 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100 disabled:opacity-40"
              title="Copy"
            >
              {copied ? <Check size={15} /> : <Copy size={15} />}
            </button>
            <button
              onClick={onClose}
              className="rounded p-1.5 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100"
              title="Close"
            >
              <X size={16} />
            </button>
          </div>
        </div>

        {files.length > 1 && (
          <div className="flex gap-1 overflow-x-auto border-b border-zinc-800 px-2 py-1.5">
            {files.map((f) => (
              <button
                key={f.path}
                onClick={() => setActive(f.path)}
                className={`flex shrink-0 items-center gap-1 rounded px-2 py-1 text-xs ${
                  f.path === active
                    ? 'bg-accent-600/15 text-accent-700'
                    : 'text-zinc-400 hover:bg-zinc-800'
                }`}
              >
                <FileText size={11} />
                {name(f.path)}
              </button>
            ))}
          </div>
        )}

        <div className="flex-1 overflow-auto p-4">
          {loading ? (
            <div className="flex justify-center py-12">
              <Spinner className="h-6 w-6" />
            </div>
          ) : error ? (
            <p className="text-sm text-red-600">{error}</p>
          ) : isMd ? (
            <div className="text-sm text-zinc-200">
              <Markdown>{content?.content ?? ''}</Markdown>
            </div>
          ) : (
            <pre className="overflow-x-auto rounded-lg border border-zinc-800 bg-zinc-950 p-3 font-mono text-xs leading-relaxed text-zinc-200">
              <code>{content?.content}</code>
            </pre>
          )}
        </div>
      </div>
    </div>
  );
}
