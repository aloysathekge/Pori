import { useEffect, useState } from 'react';
import {
  FileArchive,
  FileAudio,
  FileCode2,
  FileImage,
  FileOutput,
  FileQuestion,
  FileSpreadsheet,
  FileText,
  FileType2,
  FileVideo,
  NotebookText,
  Presentation,
  type LucideIcon,
} from 'lucide-react';
import { getFilePresentation, getStoredFileBlob } from '@/api/files';
import {
  fileExtension,
  resolveFileVisual,
  type FileVisualDescriptor,
  type FileVisualKind,
} from './fileVisualTypes';

const VISUALS: Record<FileVisualKind, { icon: LucideIcon; color: string; label: string }> = {
  pdf: { icon: FileText, color: 'text-red-400', label: 'PDF' },
  document: { icon: FileType2, color: 'text-blue-400', label: 'Document' },
  spreadsheet: { icon: FileSpreadsheet, color: 'text-emerald-400', label: 'Spreadsheet' },
  slides: { icon: Presentation, color: 'text-orange-400', label: 'Presentation' },
  image: { icon: FileImage, color: 'text-violet-400', label: 'Image' },
  video: { icon: FileVideo, color: 'text-fuchsia-400', label: 'Video' },
  audio: { icon: FileAudio, color: 'text-sky-400', label: 'Audio' },
  archive: { icon: FileArchive, color: 'text-amber-400', label: 'Archive' },
  code: { icon: FileCode2, color: 'text-cyan-400', label: 'Code' },
  markdown: { icon: NotebookText, color: 'text-indigo-400', label: 'Markdown' },
  text: { icon: FileText, color: 'text-zinc-400', label: 'Text' },
  artifact: { icon: FileOutput, color: 'text-accent-600', label: 'Artifact' },
  unknown: { icon: FileQuestion, color: 'text-zinc-500', label: 'File' },
};

export function FileTypeIcon({
  file,
  size = 18,
  className = '',
  showExtension = false,
}: {
  file: FileVisualDescriptor;
  size?: number;
  className?: string;
  showExtension?: boolean;
}) {
  const kind = resolveFileVisual(file);
  const visual = VISUALS[kind];
  const Icon = visual.icon;
  const ext = fileExtension(file.name).toUpperCase();
  return (
    <span className={`relative inline-flex shrink-0 items-center justify-center ${className}`} title={`${visual.label} file`}>
      <Icon size={size} className={visual.color} aria-hidden="true" />
      {showExtension && ext && ext.length <= 4 && (
        <span className="absolute -bottom-1 -right-1 rounded bg-zinc-950 px-0.5 font-mono text-[7px] font-bold leading-3 text-zinc-300 ring-1 ring-zinc-700">
          {ext}
        </span>
      )}
      <span className="sr-only">{visual.label}</span>
    </span>
  );
}

const MAX_LOCAL_THUMBNAIL_BYTES = 12 * 1024 * 1024;

export function FileThumbnail({
  file,
  className = 'h-12 w-16',
}: {
  file: FileVisualDescriptor;
  className?: string;
}) {
  const kind = resolveFileVisual(file);
  const fileId = file.id ?? file.file_id;
  const sizeBytes = file.size_bytes ?? file.size ?? 0;
  const supportsThumbnail = kind === 'image' || kind === 'video';
  const thumbnailKey = `${fileId ?? 'none'}:${kind}:${sizeBytes}`;
  const [thumbnail, setThumbnail] = useState<{ key: string; source: string | null; ready: boolean }>({ key: '', source: null, ready: false });
  const source = thumbnail.key === thumbnailKey ? thumbnail.source : null;
  const ready = thumbnail.key === thumbnailKey && thumbnail.ready;

  useEffect(() => {
    let cancelled = false;
    if (!supportsThumbnail || !fileId) return;
    let ownedObjectUrl: string | null = null;

    void getFilePresentation(fileId).then(async (presentation) => {
      let next = presentation.source_url;
      if (!next && sizeBytes <= MAX_LOCAL_THUMBNAIL_BYTES) {
        const blob = await getStoredFileBlob(fileId);
        next = URL.createObjectURL(blob);
        ownedObjectUrl = next;
      }
      if (cancelled) {
        if (ownedObjectUrl) URL.revokeObjectURL(ownedObjectUrl);
        return;
      }
      setThumbnail({ key: thumbnailKey, source: next, ready: false });
    }).catch(() => {
      if (!cancelled) setThumbnail({ key: thumbnailKey, source: null, ready: false });
    });

    return () => {
      cancelled = true;
      if (ownedObjectUrl) URL.revokeObjectURL(ownedObjectUrl);
    };
  }, [fileId, sizeBytes, supportsThumbnail, thumbnailKey]);

  if (!supportsThumbnail || !fileId) {
    return (
      <span className={`flex shrink-0 items-center justify-center rounded-lg border border-zinc-800 bg-zinc-900 ${className}`}>
        <FileTypeIcon file={file} size={22} showExtension />
      </span>
    );
  }

  return (
    <span className={`relative flex shrink-0 items-center justify-center overflow-hidden rounded-lg border border-zinc-800 bg-zinc-900 ${className}`}>
      {!ready && <FileTypeIcon file={file} size={22} showExtension />}
      {source && kind === 'image' && (
        <img src={source} alt="" onLoad={() => setThumbnail((current) => current.key === thumbnailKey ? { ...current, ready: true } : current)} className={`absolute inset-0 h-full w-full object-cover transition-opacity ${ready ? 'opacity-100' : 'opacity-0'}`} />
      )}
      {source && kind === 'video' && (
        <video
          src={source}
          aria-hidden="true"
          muted
          playsInline
          preload="metadata"
          onLoadedMetadata={(event) => {
            const video = event.currentTarget;
            if (Number.isFinite(video.duration) && video.duration > 0.2) video.currentTime = 0.1;
          }}
          onLoadedData={() => setThumbnail((current) => current.key === thumbnailKey ? { ...current, ready: true } : current)}
          className={`absolute inset-0 h-full w-full object-cover transition-opacity ${ready ? 'opacity-100' : 'opacity-0'}`}
        />
      )}
      {ready && kind === 'video' && <span className="absolute bottom-1 right-1 rounded bg-black/70 px-1 py-0.5 text-[8px] font-semibold text-white">VIDEO</span>}
    </span>
  );
}
