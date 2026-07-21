import { Check, LoaderCircle, Search, X } from 'lucide-react';
import { FileThumbnail } from '@/components/files/FileVisual';
import type { StoredFileReference } from '@/hooks/useAttachments';

function humanSize(bytes: number) {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${bytes} B`;
}

export function FileReferencePicker({
  files,
  loading,
  error,
  query,
  searchable,
  activeIndex,
  selectedIds,
  scopeLabel,
  onQueryChange,
  onSelect,
  onClose,
}: {
  files: StoredFileReference[];
  loading: boolean;
  error: string;
  query: string;
  searchable: boolean;
  activeIndex: number;
  selectedIds: Set<string>;
  scopeLabel: string;
  onQueryChange: (query: string) => void;
  onSelect: (file: StoredFileReference) => void;
  onClose: () => void;
}) {
  return (
    <section
      className="absolute bottom-[calc(100%+0.5rem)] left-0 z-40 flex max-h-[min(26rem,58dvh)] w-[min(28rem,calc(100vw-1rem))] flex-col overflow-hidden rounded-2xl border border-zinc-700 bg-zinc-900 shadow-2xl"
      aria-label="Choose a file to reference"
    >
      <div className="flex shrink-0 items-center justify-between gap-3 border-b border-zinc-800 px-3 py-2.5">
        <div className="min-w-0">
          <p className="text-xs font-semibold text-zinc-200">Reference a file</p>
          <p className="truncate text-[10px] text-zinc-500">{scopeLabel}</p>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200"
          aria-label="Close file picker"
        >
          <X size={15} />
        </button>
      </div>

      {searchable ? (
        <label className="relative mx-3 mt-3 block shrink-0">
          <Search size={14} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-zinc-600" />
          <input
            autoFocus
            value={query}
            onChange={(event) => onQueryChange(event.target.value)}
            placeholder="Search files"
            className="min-h-11 w-full rounded-xl border border-zinc-700 bg-zinc-950 pl-9 pr-3 text-base text-zinc-200 outline-none focus:border-accent-600 sm:text-sm"
          />
        </label>
      ) : (
        <p className="shrink-0 px-3 pt-2 text-[11px] text-zinc-500">
          {query ? <>Matching <span className="font-medium text-zinc-300">@{query}</span></> : 'Start typing a filename'}
        </p>
      )}

      <div className="min-h-0 flex-1 overflow-y-auto p-2" role="listbox">
        {loading ? (
          <div className="flex items-center justify-center gap-2 py-8 text-xs text-zinc-500">
            <LoaderCircle size={15} className="animate-spin" /> Loading files
          </div>
        ) : error ? (
          <p className="rounded-xl bg-red-500/10 px-3 py-4 text-xs text-red-400">{error}</p>
        ) : files.length === 0 ? (
          <p className="px-3 py-8 text-center text-xs leading-5 text-zinc-500">
            No matching files. Use the + button to upload one.
          </p>
        ) : (
          files.map((file, index) => {
            const selected = selectedIds.has(file.file_id);
            return (
              <button
                key={file.file_id}
                type="button"
                role="option"
                aria-selected={selected}
                onMouseDown={(event) => event.preventDefault()}
                onClick={() => onSelect(file)}
                className={`flex min-h-14 w-full items-center gap-3 rounded-xl px-3 text-left transition-colors ${index === activeIndex ? 'bg-zinc-800' : 'hover:bg-zinc-800/70'}`}
              >
                <FileThumbnail file={file} className="h-9 w-12" />
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm font-medium text-zinc-200">{file.name}</span>
                  <span className="mt-0.5 block truncate text-[10px] text-zinc-500">
                    {file.event_title ? `${file.event_title} · ` : ''}{humanSize(file.size)}
                  </span>
                </span>
                {selected && <Check size={16} className="shrink-0 text-emerald-500" />}
              </button>
            );
          })
        )}
      </div>
    </section>
  );
}
