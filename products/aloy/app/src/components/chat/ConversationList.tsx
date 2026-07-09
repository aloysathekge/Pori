import { MessageSquare, Plus, Trash2 } from 'lucide-react';
import type { ConversationResponse } from '@/types';
import { Button } from '@/components/ui/Button';
import { formatDateTime, formatRelativeTime } from '@/lib/time';

interface Props {
  conversations: ConversationResponse[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onCreate: () => void;
  onDelete: (id: string) => void;
}

export function ConversationList({
  conversations,
  activeId,
  onSelect,
  onCreate,
  onDelete,
}: Props) {
  return (
    <div className="flex h-full flex-col border-r border-zinc-800">
      <div className="flex items-center justify-between border-b border-zinc-800 px-4 py-3">
        <h2 className="text-sm font-semibold text-zinc-300">Conversations</h2>
        <Button variant="ghost" size="icon" onClick={onCreate} title="New chat">
          <Plus size={16} />
        </Button>
      </div>
      <div className="flex-1 space-y-0.5 overflow-y-auto p-2">
        {conversations.length === 0 && (
          <p className="px-3 py-6 text-center text-xs text-zinc-500">
            No conversations yet
          </p>
        )}
        {conversations.map((c) => (
          <div key={c.id} className="group relative">
            <button
              onClick={() => onSelect(c.id)}
              title={formatDateTime(c.updated_at)}
              className={`flex w-full flex-col gap-0.5 rounded-lg py-2 pl-3 pr-9 text-left text-sm transition-colors ${
                activeId === c.id
                  ? 'bg-accent-600/10 text-accent-600'
                  : 'text-zinc-300 hover:bg-zinc-800'
              }`}
            >
              <span className="flex w-full items-center gap-2">
                <MessageSquare size={13} className="shrink-0 text-zinc-500" />
                <span className="flex-1 truncate">
                  {c.title || 'New conversation'}
                </span>
              </span>
              <span className="pl-[21px] text-xs text-zinc-500">
                {formatRelativeTime(c.updated_at)}
                {c.message_count > 0 && ` · ${c.message_count} msg`}
              </span>
            </button>
            <button
              onClick={() => onDelete(c.id)}
              className="absolute right-2 top-1/2 -translate-y-1/2 opacity-0 transition-opacity group-hover:opacity-100"
              title="Delete"
            >
              <Trash2 size={14} className="text-zinc-500 hover:text-red-600" />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
