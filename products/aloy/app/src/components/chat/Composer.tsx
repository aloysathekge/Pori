import { useRef } from 'react';
import { ArrowUp, FileText, Paperclip, X } from 'lucide-react';
import type { MessageFile, MessageImage } from '@/types';

/**
 * The message composer — ONE cohesive surface (Claude/ChatGPT pattern):
 * attachment chips + auto-growing textarea + a bottom action row all live
 * inside a single rounded card. Attachments: images (the model sees them) and
 * text files (content embedded into the task). Future actions (tools, model
 * picker, voice…) join the left cluster of the action row.
 */
export function Composer({
  value,
  onChange,
  onSend,
  onAddFiles,
  pendingImages,
  onRemoveImage,
  pendingFiles,
  onRemoveFile,
  disabled,
  placeholder,
  attachFull,
}: {
  value: string;
  onChange: (v: string) => void;
  onSend: () => void;
  onAddFiles: (files: Iterable<File>) => void;
  pendingImages: MessageImage[];
  onRemoveImage: (index: number) => void;
  pendingFiles: MessageFile[];
  onRemoveFile: (index: number) => void;
  disabled: boolean;
  placeholder: string;
  attachFull: boolean;
}) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const hasAttachments = pendingImages.length > 0 || pendingFiles.length > 0;
  const canSend = !disabled && (value.trim().length > 0 || hasAttachments);

  function autoGrow() {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }

  function handleSend() {
    if (!canSend) return;
    onSend();
    requestAnimationFrame(() => {
      if (textareaRef.current) textareaRef.current.style.height = 'auto';
    });
  }

  return (
    <div
      className="rounded-2xl border border-zinc-700 bg-zinc-800 shadow-sm transition-colors focus-within:border-accent-500/70"
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault();
        onAddFiles(Array.from(e.dataTransfer.files ?? []));
      }}
    >
      {/* Attachments live inside the composer */}
      {hasAttachments && (
        <div className="flex flex-wrap items-center gap-2 px-3 pt-3">
          {pendingImages.map((img, i) => (
            <div key={`img-${i}`} className="group relative">
              <img
                src={`data:${img.media_type};base64,${img.data}`}
                alt={`attachment ${i + 1}`}
                className="h-14 w-14 rounded-lg border border-zinc-600/60 object-cover"
              />
              <button
                type="button"
                onClick={() => onRemoveImage(i)}
                className="absolute -right-1.5 -top-1.5 rounded-full bg-zinc-700 p-0.5 text-zinc-200 opacity-0 transition-opacity hover:bg-red-600 group-hover:opacity-100"
                title="Remove"
              >
                <X size={11} />
              </button>
            </div>
          ))}
          {pendingFiles.map((f, i) => (
            <div
              key={`file-${i}`}
              className="group relative flex items-center gap-2 rounded-lg border border-zinc-600/60 bg-zinc-900/60 py-2 pl-2.5 pr-3"
            >
              <FileText size={15} className="shrink-0 text-accent-600" />
              <div className="min-w-0">
                <p className="max-w-40 truncate text-xs font-medium text-zinc-200">
                  {f.name}
                </p>
                <p className="text-[10px] text-zinc-500">
                  {(f.size / 1024).toFixed(1)} KB
                </p>
              </div>
              <button
                type="button"
                onClick={() => onRemoveFile(i)}
                className="absolute -right-1.5 -top-1.5 rounded-full bg-zinc-700 p-0.5 text-zinc-200 opacity-0 transition-opacity hover:bg-red-600 group-hover:opacity-100"
                title="Remove"
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
        onChange={(e) => {
          onChange(e.target.value);
          autoGrow();
        }}
        onPaste={(e) => {
          const files = Array.from(e.clipboardData.files ?? []);
          if (files.length > 0) {
            e.preventDefault();
            onAddFiles(files);
          }
        }}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
          }
        }}
        className="max-h-[200px] w-full resize-none bg-transparent px-4 pb-1 pt-3.5 text-sm leading-relaxed text-zinc-100 placeholder-zinc-500 focus:outline-none"
      />

      {/* Action row — future actions (tools, model picker, voice…) join the
          left cluster; send stays anchored right. */}
      <div className="flex items-center justify-between px-2.5 pb-2.5">
        <div className="flex items-center gap-1">
          <input
            ref={fileInputRef}
            type="file"
            multiple
            className="hidden"
            onChange={(e) => {
              onAddFiles(Array.from(e.target.files ?? []));
              e.target.value = '';
            }}
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={disabled || attachFull}
            title="Attach images or files"
            className="rounded-lg p-2 text-zinc-500 transition-colors hover:bg-zinc-700/70 hover:text-zinc-200 disabled:opacity-40 disabled:hover:bg-transparent"
          >
            <Paperclip size={17} />
          </button>
        </div>

        <button
          type="button"
          onClick={handleSend}
          disabled={!canSend}
          title="Send"
          className={`flex h-8 w-8 items-center justify-center rounded-full transition-all ${
            canSend
              ? 'bg-accent-600 text-white shadow-sm hover:bg-accent-500'
              : 'bg-zinc-700 text-zinc-500'
          }`}
        >
          <ArrowUp size={16} strokeWidth={2.5} />
        </button>
      </div>
    </div>
  );
}
