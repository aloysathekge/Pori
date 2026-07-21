import { useEffect, useMemo, useRef, useState } from 'react';
import { ArrowUp, FileSearch, Mic, Plus, Square, Upload, X } from 'lucide-react';
import { FileTypeIcon } from '@/components/files/FileVisual';
import type { MessageFile, MessageImage } from '@/types';
import type { StoredFileReference } from '@/hooks/useAttachments';
import { FileReferencePicker } from './FileReferencePicker';

type Mention = { start: number; query: string };

const LONG_PASTE_CHARACTER_THRESHOLD = 2_000;

function pastedTextFileName(files: MessageFile[]) {
  const pastedTextPattern = /^Pasted text(?: \((\d+)\))?\.txt$/;
  let highestIndex = 0;

  for (const file of files) {
    const match = pastedTextPattern.exec(file.name);
    if (!match) continue;
    highestIndex = Math.max(highestIndex, Number(match[1] ?? 1));
  }

  return highestIndex === 0 ? 'Pasted text.txt' : `Pasted text (${highestIndex + 1}).txt`;
}

export function Composer({
  value,
  onChange,
  onSend,
  onAddFiles,
  onChooseFile,
  onSearchFiles,
  referenceFiles,
  referenceFilesLoading,
  referenceFilesError,
  referenceScopeLabel,
  pendingImages,
  onRemoveImage,
  pendingFiles,
  onRemoveFile,
  disabled,
  placeholder,
  attachFull,
  fileAttachFull,
  onStartVoice,
  onStop,
}: {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  onAddFiles: (files: Iterable<File>) => void;
  onChooseFile: (file: StoredFileReference) => void;
  onSearchFiles: (query: string) => void;
  referenceFiles: StoredFileReference[];
  referenceFilesLoading: boolean;
  referenceFilesError: string;
  referenceScopeLabel: string;
  pendingImages: MessageImage[];
  onRemoveImage: (index: number) => void;
  pendingFiles: (MessageFile & {
    uploading?: boolean;
    progress?: number;
    error?: boolean;
  })[];
  onRemoveFile: (index: number) => void;
  disabled: boolean;
  placeholder: string;
  attachFull: boolean;
  fileAttachFull: boolean;
  onStartVoice?: () => void;
  onStop?: () => void;
}) {
  const composerRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const dragDepthRef = useRef(0);
  const [menu, setMenu] = useState<'actions' | 'files' | null>(null);
  const [browserQuery, setBrowserQuery] = useState('');
  const [mention, setMention] = useState<Mention | null>(null);
  const [activeIndex, setActiveIndex] = useState(0);
  const [dragActive, setDragActive] = useState(false);

  const hasAttachments = pendingImages.length > 0 || pendingFiles.length > 0;
  const canSend = !disabled && (value.trim().length > 0 || hasAttachments);
  const pickerOpen = mention !== null || menu === 'files';
  const pickerQuery = mention?.query ?? browserQuery;
  const selectedIds = useMemo(
    () => new Set(pendingFiles.flatMap((file) => file.file_id ? [file.file_id] : [])),
    [pendingFiles],
  );

  useEffect(() => {
    if (!pickerOpen) return;
    const timer = window.setTimeout(() => onSearchFiles(pickerQuery), 120);
    return () => window.clearTimeout(timer);
  }, [onSearchFiles, pickerOpen, pickerQuery]);

  useEffect(() => {
    const element = textareaRef.current;
    if (!element) return;
    element.style.height = 'auto';
    element.style.height = `${Math.min(element.scrollHeight, 160)}px`;
  }, [value]);

  useEffect(() => {
    if (!menu && !mention) return;

    function handleOutsidePointer(event: PointerEvent) {
      if (composerRef.current?.contains(event.target as Node)) return;
      setMenu(null);
      setMention(null);
      setActiveIndex(0);
    }

    function handleEscape(event: KeyboardEvent) {
      if (event.key !== 'Escape') return;
      setMenu(null);
      setMention(null);
      setActiveIndex(0);
      textareaRef.current?.focus();
    }

    document.addEventListener('pointerdown', handleOutsidePointer);
    document.addEventListener('keydown', handleEscape);
    return () => {
      document.removeEventListener('pointerdown', handleOutsidePointer);
      document.removeEventListener('keydown', handleEscape);
    };
  }, [menu, mention]);

  function autoGrow() {
    const element = textareaRef.current;
    if (!element) return;
    element.style.height = 'auto';
    element.style.height = `${Math.min(element.scrollHeight, 160)}px`;
  }

  function closeFileUi() {
    setMenu(null);
    setMention(null);
    setActiveIndex(0);
  }

  function detectMention(text: string, cursor: number) {
    if (fileAttachFull) {
      setMention(null);
      return;
    }
    const beforeCursor = text.slice(0, cursor);
    const match = /(?:^|\s)@([^@\s]*)$/.exec(beforeCursor);
    if (!match) {
      setMention(null);
      return;
    }
    const query = match[1] ?? '';
    setMention({ start: cursor - query.length - 1, query });
    setMenu(null);
    setActiveIndex(0);
  }

  function chooseFile(file: StoredFileReference) {
    onChooseFile(file);
    if (mention) {
      const end = mention.start + mention.query.length + 1;
      const next = value.slice(0, mention.start) + value.slice(end);
      onChange(next);
      requestAnimationFrame(() => {
        textareaRef.current?.focus();
        textareaRef.current?.setSelectionRange(mention.start, mention.start);
      });
    }
    closeFileUi();
  }

  function handleSend() {
    if (!canSend) return;
    closeFileUi();
    onSend();
    requestAnimationFrame(() => {
      if (textareaRef.current) textareaRef.current.style.height = 'auto';
    });
  }

  return (
    <div
      ref={composerRef}
      className={`relative overflow-visible rounded-[1.625rem] border bg-zinc-900/95 shadow-[0_10px_32px_rgba(0,0,0,0.11)] backdrop-blur-sm transition-[border-color,box-shadow] ${dragActive ? 'border-accent-500 ring-2 ring-accent-500/20' : 'border-zinc-700/80 focus-within:border-zinc-600 focus-within:shadow-[0_12px_38px_rgba(0,0,0,0.14)]'}`}
      onDragEnter={(event) => {
        if (disabled || attachFull || !Array.from(event.dataTransfer.types).includes('Files')) return;
        event.preventDefault();
        dragDepthRef.current += 1;
        setDragActive(true);
      }}
      onDragLeave={(event) => {
        if (disabled || attachFull || !Array.from(event.dataTransfer.types).includes('Files')) return;
        event.preventDefault();
        dragDepthRef.current = Math.max(0, dragDepthRef.current - 1);
        if (dragDepthRef.current === 0) setDragActive(false);
      }}
      onDragOver={(event) => {
        if (disabled || attachFull || !Array.from(event.dataTransfer.types).includes('Files')) return;
        event.preventDefault();
        event.dataTransfer.dropEffect = 'copy';
      }}
      onDrop={(event) => {
        event.preventDefault();
        dragDepthRef.current = 0;
        setDragActive(false);
        const files = Array.from(event.dataTransfer.files ?? []);
        if (!disabled && !attachFull && files.length > 0) onAddFiles(files);
      }}
    >
      {dragActive && (
        <div className="pointer-events-none absolute inset-1 z-30 flex items-center justify-center rounded-[1.35rem] border border-dashed border-accent-500/70 bg-zinc-900/95 text-sm font-medium text-accent-600">
          Drop files to attach
        </div>
      )}

      {pickerOpen && (
        <FileReferencePicker
          files={referenceFiles}
          loading={referenceFilesLoading}
          error={referenceFilesError}
          query={pickerQuery}
          searchable={menu === 'files'}
          activeIndex={activeIndex}
          selectedIds={selectedIds}
          scopeLabel={referenceScopeLabel}
          onQueryChange={(query) => {
            setBrowserQuery(query);
            setActiveIndex(0);
          }}
          onSelect={chooseFile}
          onClose={closeFileUi}
        />
      )}

      {menu === 'actions' && (
        <div
          className="absolute bottom-[calc(100%+0.5rem)] left-0 z-40 w-60 rounded-2xl border border-zinc-700 bg-zinc-900 p-1.5 shadow-2xl"
          role="menu"
          aria-label="Add context"
        >
          <button
            type="button"
            role="menuitem"
            onClick={() => {
              setMenu(null);
              fileInputRef.current?.click();
            }}
            className="flex min-h-11 w-full items-center gap-3 rounded-xl px-3 text-left text-sm text-zinc-200 transition-colors hover:bg-zinc-800 focus-visible:bg-zinc-800 focus-visible:outline-none"
          >
            <Upload size={17} className="text-zinc-500" /> Upload from device
          </button>
          <button
            type="button"
            role="menuitem"
            onClick={() => {
              setBrowserQuery('');
              setActiveIndex(0);
              setMenu('files');
            }}
            disabled={fileAttachFull}
            className="flex min-h-11 w-full items-center gap-3 rounded-xl px-3 text-left text-sm text-zinc-200 transition-colors hover:bg-zinc-800 focus-visible:bg-zinc-800 focus-visible:outline-none disabled:cursor-not-allowed disabled:text-zinc-600 disabled:hover:bg-transparent"
          >
            <FileSearch size={17} className="text-zinc-500" />
            {fileAttachFull ? '10 files already attached' : 'Choose existing file'}
          </button>
        </div>
      )}

      {hasAttachments && (
        <div className="flex items-center gap-2 overflow-x-auto px-3 pb-1 pt-3">
          {pendingImages.map((image, index) => (
            <div key={`img-${index}`} className="group relative shrink-0">
              <img
                src={`data:${image.media_type};base64,${image.data}`}
                alt={`attachment ${index + 1}`}
                className="h-14 w-14 rounded-lg border border-zinc-600/60 object-cover"
              />
              <button
                type="button"
                onClick={() => onRemoveImage(index)}
                className="absolute -right-1.5 -top-1.5 flex h-6 w-6 items-center justify-center rounded-full bg-zinc-700 text-zinc-200 opacity-100 transition-opacity hover:bg-red-600 sm:opacity-0 sm:group-hover:opacity-100"
                aria-label={`Remove attachment ${index + 1}`}
              >
                <X size={11} />
              </button>
            </div>
          ))}
          {pendingFiles.map((file, index) => (
            <div
              key={file.file_id ?? `file-${index}`}
              className="group relative flex shrink-0 items-center gap-2 rounded-xl border border-zinc-700/80 bg-zinc-950/50 py-2 pl-2.5 pr-3"
            >
              <FileTypeIcon file={file} size={15} />
              <div className="min-w-0">
                <p className="max-w-40 truncate text-xs font-medium text-zinc-200">{file.name}</p>
                <p className={`text-[10px] ${file.error ? 'text-red-400' : 'text-zinc-500'}`}>
                  {file.uploading
                    ? `Uploading… ${file.progress ?? 0}%`
                    : file.error
                      ? 'Upload failed — not attached'
                      : file.size > 1024 * 1024
                        ? `${(file.size / (1024 * 1024)).toFixed(1)} MB`
                        : `${(file.size / 1024).toFixed(1)} KB`}
                </p>
              </div>
              <button
                type="button"
                onClick={() => onRemoveFile(index)}
                className="absolute -right-1.5 -top-1.5 flex h-6 w-6 items-center justify-center rounded-full bg-zinc-700 text-zinc-200 opacity-100 transition-opacity hover:bg-red-600 sm:opacity-0 sm:group-hover:opacity-100"
                aria-label={`Remove ${file.name}`}
              >
                <X size={11} />
              </button>
            </div>
          ))}
        </div>
      )}

      <div className="flex min-h-[3.625rem] items-end gap-1.5 px-2 py-2">
        <input
          ref={fileInputRef}
          type="file"
          multiple
          className="hidden"
          onChange={(event) => {
            onAddFiles(Array.from(event.target.files ?? []));
            event.target.value = '';
            requestAnimationFrame(() => textareaRef.current?.focus());
          }}
        />
        <button
          type="button"
          onClick={() => setMenu((current) => current === 'actions' ? null : 'actions')}
          disabled={disabled || attachFull}
          title={fileAttachFull ? '10-file limit reached' : 'Add files or reference existing files'}
          aria-label="Add context"
          aria-haspopup="menu"
          aria-expanded={menu !== null}
          className="flex h-[2.625rem] w-[2.625rem] shrink-0 items-center justify-center rounded-full text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-500/60 disabled:opacity-40 disabled:hover:bg-transparent"
        >
          <Plus size={20} strokeWidth={1.8} />
        </button>

        <textarea
          ref={textareaRef}
          rows={1}
          value={value}
          placeholder={placeholder}
          aria-label="Message Aloy"
          enterKeyHint="send"
          disabled={disabled}
          onChange={(event) => {
            const next = event.target.value;
            onChange(next);
            detectMention(next, event.target.selectionStart ?? next.length);
            autoGrow();
          }}
          onPaste={(event) => {
            const files = Array.from(event.clipboardData.files ?? []);
            if (files.length > 0) {
              event.preventDefault();
              onAddFiles(files);
              return;
            }

            const pastedText = event.clipboardData.getData('text/plain');
            if (
              !attachFull
              && !fileAttachFull
              && pastedText.length >= LONG_PASTE_CHARACTER_THRESHOLD
            ) {
              event.preventDefault();
              onAddFiles([
                new File([pastedText], pastedTextFileName(pendingFiles), {
                  type: 'text/plain',
                  lastModified: Date.now(),
                }),
              ]);
            }
          }}
          onKeyDown={(event) => {
            if (mention && referenceFiles.length > 0) {
              if (event.key === 'ArrowDown') {
                event.preventDefault();
                setActiveIndex((index) => (index + 1) % referenceFiles.length);
                return;
              }
              if (event.key === 'ArrowUp') {
                event.preventDefault();
                setActiveIndex((index) => (index - 1 + referenceFiles.length) % referenceFiles.length);
                return;
              }
              if (event.key === 'Enter') {
                event.preventDefault();
                const file = referenceFiles[activeIndex];
                if (file) chooseFile(file);
                return;
              }
            }
            if (event.key === 'Escape' && (mention || menu)) {
              event.preventDefault();
              closeFileUi();
              return;
            }
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault();
              handleSend();
            }
          }}
          className="max-h-[min(160px,30dvh)] min-h-[2.625rem] min-w-0 flex-1 resize-none bg-transparent px-1.5 py-[0.625rem] text-base leading-[1.375rem] text-zinc-100 placeholder-zinc-500 focus:outline-none"
        />

        <button
          type="button"
          onClick={onStartVoice}
          disabled={disabled || !onStartVoice}
          title={onStartVoice ? 'Start voice input' : 'Voice input coming soon'}
          aria-label={onStartVoice ? 'Start voice input' : 'Voice input coming soon'}
          className="flex h-[2.625rem] w-[2.625rem] shrink-0 items-center justify-center rounded-full text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-500/60 disabled:cursor-default disabled:opacity-80 disabled:hover:bg-transparent disabled:hover:text-zinc-500"
        >
          <Mic size={19} strokeWidth={1.8} />
        </button>

        {onStop ? (
          <button
            type="button"
            onClick={onStop}
            title="Stop generating"
            aria-label="Stop generating"
            className="flex h-[2.625rem] w-[2.625rem] shrink-0 items-center justify-center rounded-full bg-zinc-100 text-zinc-950 shadow-sm transition-colors hover:bg-zinc-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-500/60"
          >
            <Square size={13} strokeWidth={2.5} fill="currentColor" />
          </button>
        ) : (
          <button
            type="button"
            onClick={handleSend}
            disabled={!canSend}
            title="Send"
            aria-label="Send message"
            className={`flex h-[2.625rem] w-[2.625rem] shrink-0 items-center justify-center rounded-full transition-[background-color,color,transform] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-500/60 ${canSend ? 'bg-zinc-100 text-zinc-950 shadow-sm hover:bg-zinc-200 active:scale-95' : 'bg-zinc-800 text-zinc-500'}`}
          >
            <ArrowUp size={17} strokeWidth={2.5} />
          </button>
        )}
      </div>
    </div>
  );
}
