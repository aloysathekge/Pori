import { useCallback, useEffect, useState } from 'react';
import { Download, FileQuestion, MessageSquareText } from 'lucide-react';
import { getStoredFileBlob } from '@/api/files';
import type { EventFile } from '@/api/events';
import type { StoredFileReference } from '@/hooks/useAttachments';
import { Markdown } from '@/components/chat/Markdown';
import { Button } from '@/components/ui/Button';
import { Spinner } from '@/components/ui/Spinner';

interface StoredFileViewerProps {
  file: EventFile;
  onAskAloy: (reference: StoredFileReference) => void;
}

const TEXT_FILE = /(?:^text\/|json|javascript|xml|yaml|csv)/i;

function formatSize(size: number) {
  if (size >= 1024 * 1024) return `${(size / (1024 * 1024)).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(size / 1024))} KB`;
}

export function StoredFileViewer({ file, onAskAloy }: StoredFileViewerProps) {
  const [blobUrl, setBlobUrl] = useState<string | null>(null);
  const [text, setText] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const blob = await getStoredFileBlob(file.id);
      if (TEXT_FILE.test(file.content_type) || /\.(md|txt|csv|json|ya?ml|log)$/i.test(file.name)) {
        setText(await blob.text());
        setBlobUrl(null);
      } else {
        setText(null);
        setBlobUrl(URL.createObjectURL(blob));
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not open file');
      setText(null);
      setBlobUrl(null);
    } finally {
      setLoading(false);
    }
  }, [file.content_type, file.id, file.name]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- the selected file drives an authenticated blob request
    void load();
    return () => {
      setBlobUrl((current) => {
        if (current) URL.revokeObjectURL(current);
        return null;
      });
    };
  }, [load]);

  function download() {
    if (!blobUrl && text === null) return;
    const url = blobUrl ?? URL.createObjectURL(new Blob([text ?? ''], { type: file.content_type }));
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = file.name;
    anchor.click();
    if (!blobUrl) URL.revokeObjectURL(url);
  }

  const isMarkdown = /\.md$/i.test(file.name) || file.content_type === 'text/markdown';
  const isImage = file.content_type.startsWith('image/');
  const isPdf = file.content_type === 'application/pdf';

  return (
    <section className="flex h-full min-h-0 flex-col bg-zinc-950">
      <div className="flex min-h-11 shrink-0 items-center justify-between gap-3 border-b border-zinc-800 px-3">
        <p className="truncate text-xs text-zinc-500">{file.kind} · {formatSize(file.size_bytes)} · {file.content_type}</p>
        <div className="flex shrink-0 items-center gap-1">
          <Button size="sm" variant="ghost" onClick={() => onAskAloy({ file_id: file.id, name: file.name, size: file.size_bytes })}>
            <MessageSquareText size={14} /> Ask Aloy
          </Button>
          <button type="button" onClick={download} disabled={!blobUrl && text === null} className="rounded-md p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-40" title="Download file"><Download size={15} /></button>
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-auto">
        {loading ? (
          <div className="flex h-full items-center justify-center"><Spinner className="h-6 w-6" /></div>
        ) : error ? (
          <div className="flex h-full items-center justify-center px-6 text-center"><div><p className="text-sm text-red-600">{error}</p><Button size="sm" variant="ghost" className="mt-3" onClick={() => void load()}>Try again</Button></div></div>
        ) : isMarkdown && text !== null ? (
          <article className="mx-auto max-w-3xl p-6 text-sm text-zinc-200"><Markdown>{text}</Markdown></article>
        ) : text !== null ? (
          <pre className="m-4 min-h-[calc(100%-2rem)] overflow-x-auto rounded-xl border border-zinc-800 bg-zinc-900 p-4 font-mono text-xs leading-6 text-zinc-200"><code>{text}</code></pre>
        ) : isImage && blobUrl ? (
          <div className="flex min-h-full items-center justify-center p-5"><img src={blobUrl} alt={file.name} className="max-h-full max-w-full rounded-lg object-contain shadow-sm" /></div>
        ) : isPdf && blobUrl ? (
          <iframe src={blobUrl} sandbox="" title={file.name} className="h-full w-full border-0 bg-white" />
        ) : (
          <div className="flex h-full items-center justify-center px-8 text-center">
            <div className="max-w-sm"><FileQuestion size={30} className="mx-auto text-zinc-500" /><h2 className="mt-4 font-display text-base font-semibold text-zinc-200">Preview is not available for this format</h2><p className="mt-2 text-sm leading-6 text-zinc-500">Ask Aloy to inspect the trusted file, or download it to open in its native application.</p></div>
          </div>
        )}
      </div>
    </section>
  );
}
