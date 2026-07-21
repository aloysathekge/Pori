import { useEffect, useMemo, useRef, useState } from 'react';
import { ArrowUp, FileSearch, Plus, Square, Upload, X } from 'lucide-react';
import { FileTypeIcon } from '@/components/files/FileVisual';
import type { MessageFile, MessageImage } from '@/types';
import type { StoredFileReference } from '@/hooks/useAttachments';
import { FileReferencePicker } from './FileReferencePicker';

type Mention = { start: number; query: string };

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
  onStop?: () => void;
}) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [menu, setMenu] = useState<'actions' | 'files' | null>(null);
  const [browserQuery, setBrowserQuery] = useState('');
  const [mention, setMention] = useState<Mention | null>(null);
  const [activeIndex, setActiveIndex] = useState(0);

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

  function autoGrow() {
    const element = textareaRef.current;
    if (!element) return;
    element.style.height = 'auto';
    element.style.height = `${Math.min(element.scrollHeight, 200)}px`;
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
      className="relative rounded-2xl border border-zinc-700 bg-zinc-800 shadow-sm transition-colors focus-within:border-accent-500/70"
      onDragOver={(event) => event.preventDefault()}
      onDrop={(event) => {
        event.preventDefault();
        onAddFiles(Array.from(event.dataTransfer.files ?? []));
      }}
    >
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
        <div className="absolute bottom-12 left-2 z-40 w-56 rounded-2xl border border-zinc-700 bg-zinc-900 p-1.5 shadow-2xl">
          <button
            type="button"
            onClick={() => {
              setMenu(null);
              fileInputRef.current?.click();
            }}
            className="flex min-h-12 w-full items-center gap-3 rounded-xl px-3 text-left text-sm text-zinc-200 hover:bg-zinc-800"
          >
            <Upload size={17} className="text-zinc-500" /> Upload from device
          </button>
          <button
            type="button"
            onClick={() => {
              setBrowserQuery('');
              setActiveIndex(0);
              setMenu('files');
            }}
            disabled={fileAttachFull}
            className="flex min-h-12 w-full items-center gap-3 rounded-xl px-3 text-left text-sm text-zinc-200 hover:bg-zinc-800 disabled:cursor-not-allowed disabled:text-zinc-600 disabled:hover:bg-transparent"
          >
            <FileSearch size={17} className="text-zinc-500" />
            {fileAttachFull ? '10 files already attached' : 'Choose existing file'}
          </button>
        </div>
      )}

      {hasAttachments && (
        <div className="flex flex-wrap items-center gap-2 px-3 pt-3">
          {pendingImages.map((image, index) => (
            <div key={`img-${index}`} className="group relative">
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
              className="group relative flex items-center gap-2 rounded-lg border border-zinc-600/60 bg-zinc-900/60 py-2 pl-2.5 pr-3"
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

      <textarea
        ref={textareaRef}
        rows={1}
        value={value}
        placeholder={placeholder}
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
        className="max-h-[min(200px,32dvh)] w-full resize-none bg-transparent px-4 pb-1 pt-3.5 text-base leading-relaxed text-zinc-100 placeholder-zinc-500 focus:outline-none sm:text-sm"
      />

      <div className="flex items-center justify-between px-2.5 pb-2.5">
        <div className="flex items-center gap-1">
          <input
            ref={fileInputRef}
            type="file"
            multiple
            className="hidden"
            onChange={(event) => {
              onAddFiles(Array.from(event.target.files ?? []));
              event.target.value = '';
            }}
          />
          <button
            type="button"
            onClick={() => setMenu((current) => current === 'actions' ? null : 'actions')}
            disabled={disabled || attachFull}
            title="Add files or reference existing files"
            aria-label="Add context"
            aria-expanded={menu !== null}
            className="flex h-11 w-11 items-center justify-center rounded-xl text-zinc-500 transition-colors hover:bg-zinc-700/70 hover:text-zinc-200 disabled:opacity-40 disabled:hover:bg-transparent sm:h-9 sm:w-9"
          >
            <Plus size={18} />
          </button>
          <span className="hidden text-[10px] text-zinc-600 sm:inline">
            {fileAttachFull ? '10-file limit reached' : 'Type @ to reference a file'}
          </span>
        </div>

        {onStop ? (
          <button
            type="button"
            onClick={onStop}
            title="Stop generating"
            className="flex h-11 w-11 items-center justify-center rounded-full bg-accent-600 text-white shadow-sm transition-all hover:bg-accent-500 sm:h-8 sm:w-8"
          >
            <Square size={13} strokeWidth={2.5} fill="currentColor" />
          </button>
        ) : (
          <button
            type="button"
            onClick={handleSend}
            disabled={!canSend}
            title="Send"
            className={`flex h-11 w-11 items-center justify-center rounded-full transition-all sm:h-8 sm:w-8 ${canSend ? 'bg-accent-600 text-white shadow-sm hover:bg-accent-500' : 'bg-zinc-700 text-zinc-500'}`}
          >
            <ArrowUp size={16} strokeWidth={2.5} />
          </button>
        )}
      </div>
    </div>
  );
}
