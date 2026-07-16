import { useCallback, useEffect, useState } from 'react';
import { Check, Copy, Download, MessageSquareText } from 'lucide-react';
import { getArtifactContent, type ArtifactContent } from '@/api/artifacts';
import type { StoredFileReference } from '@/hooks/useAttachments';
import { Button } from '@/components/ui/Button';
import { Spinner } from '@/components/ui/Spinner';
import { Markdown } from '@/components/chat/Markdown';

interface ArtifactViewerProps {
  conversationId: string;
  path: string;
  onAskAloy: (reference: StoredFileReference) => void;
}

function fileName(path: string) {
  return path.split(/[/\\]/).pop() || path;
}

export function ArtifactViewer({ conversationId, path, onAskAloy }: ArtifactViewerProps) {
  const [content, setContent] = useState<ArtifactContent | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [copied, setCopied] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      setContent(await getArtifactContent(conversationId, path));
    } catch (cause) {
      setContent(null);
      setError(cause instanceof Error ? cause.message : 'Could not open artifact');
    } finally {
      setLoading(false);
    }
  }, [conversationId, path]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- the selected tab drives the artifact request
    void load();
  }, [load]);

  function copy() {
    if (!content) return;
    void navigator.clipboard.writeText(content.content).then(() => {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    });
  }

  function download() {
    if (!content) return;
    const url = URL.createObjectURL(new Blob([content.content], { type: 'text/plain' }));
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = fileName(path);
    anchor.click();
    URL.revokeObjectURL(url);
  }

  return (
    <section className="flex h-full min-h-0 flex-col bg-zinc-950">
      <div className="flex min-h-11 shrink-0 items-center justify-between gap-3 border-b border-zinc-800 px-3">
        <div className="min-w-0">
          <p className="truncate font-mono text-[11px] text-zinc-500">{path}</p>
        </div>
        <div className="flex shrink-0 items-center gap-1">
          <Button
            size="sm"
            variant="ghost"
            disabled={!content}
            onClick={() => content && onAskAloy({
              file_id: content.file_id,
              name: fileName(path),
              size: content.content.length,
            })}
          >
            <MessageSquareText size={14} /> Ask Aloy
          </Button>
          <button type="button" onClick={copy} disabled={!content} className="rounded-md p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-40" title="Copy artifact">
            {copied ? <Check size={15} /> : <Copy size={15} />}
          </button>
          <button type="button" onClick={download} disabled={!content} className="rounded-md p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-40" title="Download artifact">
            <Download size={15} />
          </button>
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-auto p-5">
        {loading ? (
          <div className="flex h-full items-center justify-center"><Spinner className="h-6 w-6" /></div>
        ) : error ? (
          <div className="flex h-full items-center justify-center px-6 text-center">
            <div><p className="text-sm text-red-600">{error}</p><Button size="sm" variant="ghost" className="mt-3" onClick={() => void load()}>Try again</Button></div>
          </div>
        ) : content?.language === 'markdown' ? (
          <article className="mx-auto max-w-3xl text-sm text-zinc-200"><Markdown>{content.content}</Markdown></article>
        ) : (
          <pre className="min-h-full overflow-x-auto rounded-xl border border-zinc-800 bg-zinc-900 p-4 font-mono text-xs leading-6 text-zinc-200"><code>{content?.content}</code></pre>
        )}
      </div>
    </section>
  );
}
