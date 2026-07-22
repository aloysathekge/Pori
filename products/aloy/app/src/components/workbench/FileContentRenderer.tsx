import { useMemo, useState } from 'react';
import { Code2, Eye, FileQuestion, FileWarning, Music2, Presentation, ShieldCheck } from 'lucide-react';
import type { FilePresentation } from '@/api/files';
import { Markdown } from '@/components/chat/Markdown';
import {
  buildHtmlArtifactPreviewDocument,
  HTML_ARTIFACT_IFRAME_SANDBOX,
} from './htmlArtifactSecurity';

interface FileContentRendererProps {
  presentation: FilePresentation;
  sourceUrl: string | null;
  text: string | null;
}

function Unavailable({ presentation }: { presentation: FilePresentation }) {
  return (
    <div className="flex h-full items-center justify-center px-8 text-center">
      <div className="max-w-md">
        <FileQuestion size={30} className="mx-auto text-zinc-500" />
        <h2 className="mt-4 font-display text-base font-semibold text-zinc-200">
          Preview is not available for this format
        </h2>
        <p className="mt-2 text-sm leading-6 text-zinc-500">
          The original file is safe and unchanged. Download it or ask Aloy to inspect its trusted reference.
        </p>
        {presentation.preview_error && (
          <p className="mt-3 rounded-lg border border-amber-500/20 bg-amber-500/8 px-3 py-2 text-xs text-amber-500">
            {presentation.preview_error}
          </p>
        )}
      </div>
    </div>
  );
}

function DocumentPreview({ presentation }: { presentation: FilePresentation }) {
  const blocks = presentation.preview?.blocks ?? [];
  if (!blocks.length) return <Unavailable presentation={presentation} />;
  return (
    <article className="mx-auto max-w-3xl px-6 py-8 text-[15px] leading-7 text-zinc-200 sm:px-10">
      {blocks.map((block, index) => (
        <p key={`${index}:${block.slice(0, 24)}`} className="mb-4 whitespace-pre-wrap">{block}</p>
      ))}
      {presentation.preview?.truncated && <PreviewTruncated />}
    </article>
  );
}

function SpreadsheetPreview({ presentation }: { presentation: FilePresentation }) {
  const sheets = presentation.preview?.sheets ?? [];
  const [activeIndex, setActiveIndex] = useState(0);
  if (!sheets.length) return <Unavailable presentation={presentation} />;
  const active = sheets[Math.min(activeIndex, sheets.length - 1)]!;
  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex shrink-0 gap-1 overflow-x-auto border-b border-zinc-800 px-3 py-2">
        {sheets.map((sheet, index) => (
          <button
            key={`${index}:${sheet.name}`}
            type="button"
            onClick={() => setActiveIndex(index)}
            className={`shrink-0 rounded-md px-3 py-1.5 text-xs ${index === activeIndex ? 'bg-accent-600/15 text-accent-500' : 'text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200'}`}
          >
            {sheet.name}
          </button>
        ))}
      </div>
      <div className="min-h-0 flex-1 overflow-auto p-3">
        <table className="min-w-full border-separate border-spacing-0 text-left text-xs text-zinc-300">
          <tbody>
            {active.rows.map((row, rowIndex) => (
              <tr key={rowIndex}>
                <th className="sticky left-0 border-b border-r border-zinc-800 bg-zinc-900 px-2 py-1.5 text-right font-mono font-normal text-zinc-600">{rowIndex + 1}</th>
                {row.map((cell, cellIndex) => (
                  <td key={cellIndex} className="max-w-80 whitespace-pre-wrap border-b border-r border-zinc-800 px-2.5 py-1.5 align-top">{cell}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        {presentation.preview?.truncated && <PreviewTruncated />}
      </div>
    </div>
  );
}

function SlidesPreview({ presentation }: { presentation: FilePresentation }) {
  const slides = presentation.preview?.slides ?? [];
  if (!slides.length) return <Unavailable presentation={presentation} />;
  return (
    <div className="mx-auto grid max-w-6xl gap-5 p-5 md:grid-cols-2">
      {slides.map((slide) => (
        <section key={slide.number} className="aspect-[16/9] overflow-auto rounded-xl border border-zinc-700 bg-zinc-900 p-5 shadow-lg">
          <div className="mb-4 flex items-center gap-2 text-xs text-zinc-500"><Presentation size={14} /> Slide {slide.number}</div>
          <p className="whitespace-pre-wrap text-sm leading-6 text-zinc-200">{slide.text || 'No extractable text on this slide.'}</p>
        </section>
      ))}
      {presentation.preview?.truncated && <PreviewTruncated />}
    </div>
  );
}

function PreviewTruncated() {
  return (
    <div className="m-4 flex items-center gap-2 rounded-lg border border-amber-500/20 bg-amber-500/8 px-3 py-2 text-xs text-amber-500">
      <FileWarning size={14} /> This trusted preview is bounded. Download the original to inspect the remainder.
    </div>
  );
}

function HtmlPreview({ presentation, text }: { presentation: FilePresentation; text: string | null }) {
  const canPreview = text !== null && !presentation.preview?.truncated;
  const [mode, setMode] = useState<'preview' | 'code'>(() =>
    presentation.kind === 'artifact' && canPreview ? 'preview' : 'code',
  );
  const document = useMemo(
    () => text === null ? '' : buildHtmlArtifactPreviewDocument(text),
    [text],
  );

  if (text === null) return <Unavailable presentation={presentation} />;

  return (
    <div className="flex h-full min-h-[32rem] flex-col">
      <div className="flex min-h-11 shrink-0 items-center justify-between gap-3 border-b border-zinc-800 bg-zinc-950 px-3">
        <div className="flex rounded-lg border border-zinc-800 bg-zinc-900 p-0.5">
          <button
            type="button"
            onClick={() => setMode('preview')}
            disabled={!canPreview}
            className={`flex min-h-8 items-center gap-1.5 rounded-md px-3 text-xs font-medium ${mode === 'preview' ? 'bg-zinc-700 text-zinc-100' : 'text-zinc-500 hover:text-zinc-200'} disabled:cursor-not-allowed disabled:opacity-40`}
          >
            <Eye size={14} /> Preview
          </button>
          <button
            type="button"
            onClick={() => setMode('code')}
            className={`flex min-h-8 items-center gap-1.5 rounded-md px-3 text-xs font-medium ${mode === 'code' ? 'bg-zinc-700 text-zinc-100' : 'text-zinc-500 hover:text-zinc-200'}`}
          >
            <Code2 size={14} /> Code
          </button>
        </div>
        <p className="flex min-w-0 items-center gap-1.5 truncate text-[11px] text-zinc-500" title="Runs without network, Aloy, Event state, storage, navigation, or download access">
          <ShieldCheck size={13} className="shrink-0 text-emerald-600" /> Isolated preview
        </p>
      </div>

      {mode === 'preview' && canPreview ? (
        <iframe
          key={presentation.sha256}
          title={`Interactive preview of ${presentation.name}`}
          srcDoc={document}
          sandbox={HTML_ARTIFACT_IFRAME_SANDBOX}
          referrerPolicy="no-referrer"
          allow="camera 'none'; geolocation 'none'; microphone 'none'"
          className="min-h-0 flex-1 border-0 bg-white"
        />
      ) : (
        <div className="min-h-0 flex-1 overflow-auto p-4">
          <pre className="min-h-full overflow-x-auto rounded-xl border border-zinc-800 bg-zinc-900 p-4 font-mono text-xs leading-6 text-zinc-200"><code>{text}</code></pre>
          {presentation.preview?.truncated && (
            <div className="mt-3"><PreviewTruncated /></div>
          )}
        </div>
      )}
    </div>
  );
}

export function FileContentRenderer({ presentation, sourceUrl, text }: FileContentRendererProps) {
  switch (presentation.renderer) {
    case 'markdown':
      return text === null ? <Unavailable presentation={presentation} /> : (
        <>
          <article className="mx-auto max-w-3xl p-6 text-sm text-zinc-200"><Markdown>{text}</Markdown></article>
          {presentation.preview?.truncated && <PreviewTruncated />}
        </>
      );
    case 'html':
      return <HtmlPreview presentation={presentation} text={text} />;
    case 'code':
    case 'text':
      return text === null ? <Unavailable presentation={presentation} /> : (
        <div className="min-h-full p-4">
          <pre className="min-h-full overflow-x-auto rounded-xl border border-zinc-800 bg-zinc-900 p-4 font-mono text-xs leading-6 text-zinc-200"><code>{text}</code></pre>
          {presentation.preview?.truncated && <PreviewTruncated />}
        </div>
      );
    case 'image':
      return sourceUrl ? (
        <div className="flex min-h-full items-center justify-center p-5"><img src={sourceUrl} alt={presentation.name} className="max-h-full max-w-full rounded-lg object-contain shadow-sm" /></div>
      ) : <Unavailable presentation={presentation} />;
    case 'pdf':
      return sourceUrl ? (
        <object data={sourceUrl} type="application/pdf" aria-label={presentation.name} className="h-full min-h-[32rem] w-full bg-white">
          <Unavailable presentation={presentation} />
        </object>
      ) : <Unavailable presentation={presentation} />;
    case 'video':
      return sourceUrl ? (
        <div className="flex min-h-full items-center justify-center bg-black p-3"><video src={sourceUrl} controls playsInline preload="metadata" className="max-h-full max-w-full" aria-label={presentation.name} /></div>
      ) : <Unavailable presentation={presentation} />;
    case 'audio':
      return sourceUrl ? (
        <div className="flex min-h-full items-center justify-center px-6"><div className="w-full max-w-2xl rounded-2xl border border-zinc-800 bg-zinc-900 p-6"><Music2 size={28} className="mb-5 text-accent-600" /><p className="mb-4 truncate text-sm text-zinc-300">{presentation.name}</p><audio src={sourceUrl} controls preload="metadata" className="w-full" aria-label={presentation.name} /></div></div>
      ) : <Unavailable presentation={presentation} />;
    case 'document':
      return <DocumentPreview presentation={presentation} />;
    case 'spreadsheet':
      return <SpreadsheetPreview presentation={presentation} />;
    case 'slides':
      return <SlidesPreview presentation={presentation} />;
    default:
      return <Unavailable presentation={presentation} />;
  }
}
