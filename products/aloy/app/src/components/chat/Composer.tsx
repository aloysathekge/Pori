import { useRef } from 'react';
import { ArrowUp, ImagePlus, X } from 'lucide-react';
import type { MessageImage } from '@/types';

/**
 * The message composer — ONE cohesive surface (Claude/ChatGPT pattern):
 * image chips + auto-growing textarea + a bottom action row all live inside a
 * single rounded card, so attach (and future actions: tools, model picker,
 * voice…) read as part of the composer, not floating beside it. Add new
 * actions to the left cluster of the action row.
 */
export function Composer({
  value,
  onChange,
  onSend,
  onAddImages,
  pendingImages,
  onRemoveImage,
  disabled,
  placeholder,
  maxImages,
}: {
  value: string;
  onChange: (v: string) => void;
  onSend: () => void;
  onAddImages: (files: Iterable<File>) => void;
  pendingImages: MessageImage[];
  onRemoveImage: (index: number) => void;
  disabled: boolean;
  placeholder: string;
  maxImages: number;
}) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const canSend = !disabled && (value.trim().length > 0 || pendingImages.length > 0);

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
      className="rounded-2xl border border-zinc-700 bg-zinc-800 shadow-sm transition-colors focus-within:border-accent-500 focus-within:ring-1 focus-within:ring-accent-500/40"
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault();
        onAddImages(Array.from(e.dataTransfer.files ?? []));
      }}
    >
      {/* Attached images live inside the composer */}
      {pendingImages.length > 0 && (
        <div className="flex gap-2 px-3 pt-3">
          {pendingImages.map((img, i) => (
            <div key={i} className="group relative">
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
          if (files.some((f) => f.type.startsWith('image/'))) {
            e.preventDefault();
            onAddImages(files);
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
            accept="image/png,image/jpeg,image/gif,image/webp"
            multiple
            className="hidden"
            onChange={(e) => {
              onAddImages(Array.from(e.target.files ?? []));
              e.target.value = '';
            }}
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={disabled || pendingImages.length >= maxImages}
            title="Attach image"
            className="rounded-lg p-2 text-zinc-500 transition-colors hover:bg-zinc-700/70 hover:text-zinc-200 disabled:opacity-40 disabled:hover:bg-transparent"
          >
            <ImagePlus size={17} />
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
