import { useEffect, useMemo, useRef, useState } from 'react';
import { Check, ChevronDown, MessageSquare, Plus, Search, Trash2 } from 'lucide-react';
import type { ConversationResponse } from '@/types';
import { formatRelativeTime } from '@/lib/time';

/**
 * Zero-footprint conversation switcher: the current conversation's title is a
 * dropdown in the top bar; the full (searchable) list lives in a popover, so
 * no horizontal screen space is ever reserved for navigation.
 */
export function ConversationSwitcher({
  conversations,
  activeId,
  onSelect,
  onCreate,
  onDelete,
}: {
  conversations: ConversationResponse[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onCreate: () => void;
  onDelete: (id: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const rootRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  const active = conversations.find((c) => c.id === activeId);
  const title = active?.title || (activeId ? 'New conversation' : 'Conversations');

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return conversations;
    return conversations.filter((c) =>
      (c.title || 'New conversation').toLowerCase().includes(q),
    );
  }, [conversations, query]);

  // Close on outside click / Escape; focus search when opened.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);
    searchRef.current?.focus();
    return () => {
      document.removeEventListener('mousedown', onDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  function choose(id: string) {
    setOpen(false);
    setQuery('');
    onSelect(id);
  }

  return (
    <div ref={rootRef} className="relative min-w-0">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex min-h-11 max-w-[55vw] items-center gap-2 rounded-lg px-2.5 py-1.5 text-sm font-medium text-zinc-200 transition-colors hover:bg-zinc-800 sm:min-h-0 sm:max-w-[60vw]"
        title="Switch conversation"
      >
        <MessageSquare size={15} className="shrink-0 text-zinc-500" />
        <span className="truncate">{title}</span>
        <ChevronDown
          size={14}
          className={`shrink-0 text-zinc-500 transition-transform ${
            open ? 'rotate-180' : ''
          }`}
        />
      </button>

      {open && (
        <div className="absolute left-0 top-full z-40 mt-1.5 w-[min(20rem,calc(100vw-1rem))] overflow-hidden rounded-xl border border-zinc-700/70 bg-zinc-900 shadow-2xl">
          <div className="flex items-center gap-2 border-b border-zinc-800 px-3 py-2">
            <Search size={14} className="shrink-0 text-zinc-500" />
            <input
              ref={searchRef}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search conversations…"
              className="w-full bg-transparent text-sm text-zinc-100 placeholder-zinc-500 focus:outline-none"
            />
          </div>

          <div className="max-h-80 overflow-y-auto p-1.5">
            {filtered.length === 0 ? (
              <p className="px-3 py-6 text-center text-xs text-zinc-500">
                No conversations found
              </p>
            ) : (
              filtered.map((c) => (
                <div key={c.id} className="group relative">
                  <button
                    type="button"
                    onClick={() => choose(c.id)}
                    className={`flex w-full flex-col gap-0.5 rounded-lg py-2 pl-3 pr-14 text-left text-sm transition-colors ${
                      c.id === activeId
                        ? 'bg-accent-600/10 text-accent-600'
                        : 'text-zinc-300 hover:bg-zinc-800'
                    }`}
                  >
                    <span className="truncate">
                      {c.title || 'New conversation'}
                    </span>
                    <span className="text-xs text-zinc-500">
                      {formatRelativeTime(c.updated_at)}
                      {c.message_count > 0 && ` · ${c.message_count} msg`}
                    </span>
                  </button>
                  {c.id === activeId && (
                    <Check
                      size={14}
                      className="absolute right-9 top-1/2 -translate-y-1/2 text-accent-600"
                    />
                  )}
                  <button
                    type="button"
                    onClick={() => onDelete(c.id)}
                    className="absolute right-2.5 top-1/2 -translate-y-1/2 text-zinc-600 opacity-0 transition-opacity hover:text-red-500 group-hover:opacity-100"
                    title="Delete"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              ))
            )}
          </div>

          <button
            type="button"
            onClick={() => {
              setOpen(false);
              setQuery('');
              onCreate();
            }}
            className="flex w-full items-center gap-2 border-t border-zinc-800 px-3.5 py-2.5 text-sm text-zinc-300 transition-colors hover:bg-zinc-800 hover:text-accent-600"
          >
            <Plus size={15} /> New conversation
          </button>
        </div>
      )}
    </div>
  );
}
