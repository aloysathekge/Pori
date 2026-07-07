import { MessageSquare, Plus, Trash2 } from 'lucide-react';
import type { ConversationResponse } from '@/types';
import { Button } from '@/components/ui/Button';

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
          <button
            key={c.id}
            onClick={() => onSelect(c.id)}
            className={`group flex w-full items-center gap-2 rounded-lg px-3 py-2.5 text-left text-sm transition-colors ${
              activeId === c.id
                ? 'bg-accent-600/10 text-accent-400'
                : 'text-zinc-400 hover:bg-zinc-800'
            }`}
          >
            <MessageSquare size={14} className="shrink-0" />
            <span className="flex-1 truncate">
              {c.title || 'New conversation'}
            </span>
            <span className="shrink-0 text-xs text-zinc-600">
              {c.message_count}
            </span>
            <button
              onClick={(e) => {
                e.stopPropagation();
                onDelete(c.id);
              }}
              className="shrink-0 opacity-0 transition-opacity group-hover:opacity-100"
              title="Delete"
            >
              <Trash2 size={14} className="text-zinc-500 hover:text-red-400" />
            </button>
          </button>
        ))}
      </div>
    </div>
  );
}
