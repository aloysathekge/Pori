import { useCallback, useEffect, useRef, useState } from 'react';
import { Download, MessageSquareText } from 'lucide-react';
import { getFilePresentation, getStoredFileBlob, type FilePresentation } from '@/api/files';
import type { EventFile } from '@/api/events';
import type { StoredFileReference } from '@/hooks/useAttachments';
import { Button } from '@/components/ui/Button';
import { Spinner } from '@/components/ui/Spinner';
import { FileContentRenderer } from './FileContentRenderer';
import { FileTypeIcon } from '@/components/files/FileVisual';

interface StoredFileViewerProps {
  file: EventFile;
  onAskAloy?: (reference: StoredFileReference) => void;
}

const SOURCE_RENDERERS = new Set(['image', 'pdf', 'audio', 'video']);
const TEXT_RENDERERS = new Set(['markdown', 'text']);

function formatSize(size: number) {
  if (size >= 1024 * 1024) return `${(size / (1024 * 1024)).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(size / 1024))} KB`;
}

function rendererLabel(renderer: FilePresentation['renderer']) {
  return ({
    markdown: 'Markdown',
    text: 'Text',
    image: 'Image',
    pdf: 'PDF',
    audio: 'Audio',
    video: 'Video',
    document: 'Document',
    spreadsheet: 'Workbook',
    slides: 'Presentation',
    unknown: 'File',
  })[renderer];
}

export function StoredFileViewer({ file, onAskAloy }: StoredFileViewerProps) {
  const [presentation, setPresentation] = useState<FilePresentation | null>(null);
  const [sourceUrl, setSourceUrl] = useState<string | null>(null);
  const [text, setText] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState('');
  const objectUrlRef = useRef<string | null>(null);
  const requestRef = useRef(0);

  const revokeObjectUrl = useCallback(() => {
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    objectUrlRef.current = null;
  }, []);

  const load = useCallback(async () => {
    const request = ++requestRef.current;
    setLoading(true);
    setError('');
    revokeObjectUrl();
    try {
      const next = await getFilePresentation(file.id);
      let nextSource = next.source_url;
      let nextText: string | null = null;
      if (TEXT_RENDERERS.has(next.renderer)) {
        nextText = await (await getStoredFileBlob(file.id)).text();
      } else if (SOURCE_RENDERERS.has(next.renderer) && !nextSource) {
        const blob = await getStoredFileBlob(file.id);
        nextSource = URL.createObjectURL(blob);
        objectUrlRef.current = nextSource;
      }
      if (request !== requestRef.current) {
        if (objectUrlRef.current === nextSource) revokeObjectUrl();
        return;
      }
      setPresentation(next);
      setSourceUrl(nextSource);
      setText(nextText);
    } catch (cause) {
      if (request !== requestRef.current) return;
      setPresentation(null);
      setSourceUrl(null);
      setText(null);
      setError(cause instanceof Error ? cause.message : 'Could not open file');
    } finally {
      if (request === requestRef.current) setLoading(false);
    }
  }, [file.id, revokeObjectUrl]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- selected file drives authenticated presentation requests
    void load();
    return () => {
      requestRef.current += 1;
      revokeObjectUrl();
    };
  }, [load, revokeObjectUrl]);

  async function download() {
    setDownloading(true);
    try {
      const blob = await getStoredFileBlob(file.id);
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = file.name;
      anchor.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 0);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not download file');
    } finally {
      setDownloading(false);
    }
  }

  return (
    <section className="flex h-full min-h-0 flex-col bg-zinc-950">
      <div className="flex min-h-11 shrink-0 items-center justify-between gap-3 border-b border-zinc-800 px-3">
        <div className="flex min-w-0 items-center gap-2">
          <FileTypeIcon file={file} size={16} />
          <p className="truncate text-xs text-zinc-500">
            {presentation ? rendererLabel(presentation.renderer) : file.kind} · {formatSize(file.size_bytes)} · {file.content_type}
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-1">
          {onAskAloy && (
            <Button size="sm" variant="ghost" onClick={() => onAskAloy({ file_id: file.id, name: file.name, size: file.size_bytes })}>
              <MessageSquareText size={14} /> Ask Aloy
            </Button>
          )}
          <button type="button" onClick={() => void download()} disabled={downloading} className="rounded-md p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-40" title="Download file" aria-label={`Download ${file.name}`}>
            {downloading ? <Spinner className="h-[15px] w-[15px]" /> : <Download size={15} />}
          </button>
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-auto">
        {loading ? (
          <div className="flex h-full items-center justify-center"><Spinner className="h-6 w-6" /></div>
        ) : error ? (
          <div className="flex h-full items-center justify-center px-6 text-center"><div><p className="text-sm text-red-600">{error}</p><Button size="sm" variant="ghost" className="mt-3" onClick={() => void load()}>Try again</Button></div></div>
        ) : presentation ? (
          <FileContentRenderer key={presentation.file_id} presentation={presentation} sourceUrl={sourceUrl} text={text} />
        ) : null}
      </div>
    </section>
  );
}
