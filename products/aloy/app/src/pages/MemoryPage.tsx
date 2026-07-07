import { useState, useEffect } from 'react';
import { Brain, Search, Trash2, RotateCcw, Save, Plus } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { TextArea } from '@/components/ui/TextArea';
import { Input } from '@/components/ui/Input';
import { Badge } from '@/components/ui/Badge';
import { Spinner } from '@/components/ui/Spinner';
import {
  getMemory,
  updateMemoryBlock,
  resetMemory,
  listKnowledgeEntries,
  createKnowledgeEntry,
  searchKnowledgeEntries,
  deleteKnowledgeEntry,
} from '@/api/memory';
import type { MemoryBlock, KnowledgeEntry } from '@/types';

export function MemoryPage() {
  const [blocks, setBlocks] = useState<MemoryBlock[]>([]);
  const [editedBlocks, setEditedBlocks] = useState<Record<string, string>>({});
  const [knowledge, setKnowledge] = useState<KnowledgeEntry[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [newEntry, setNewEntry] = useState('');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState<string | null>(null);
  const [searching, setSearching] = useState(false);
  const [adding, setAdding] = useState(false);

  useEffect(() => {
    load();
  }, []);

  async function load() {
    setLoading(true);
    try {
      const [mem, entries] = await Promise.all([
        getMemory(),
        listKnowledgeEntries(),
      ]);
      setBlocks(mem.blocks);
      setKnowledge(entries);
      setEditedBlocks({});
    } catch {
      // handle silently
    } finally {
      setLoading(false);
    }
  }

  async function saveBlock(label: string) {
    const value = editedBlocks[label];
    if (value === undefined) return;
    setSaving(label);
    try {
      await updateMemoryBlock(label, value);
      setBlocks((prev) =>
        prev.map((b) => (b.label === label ? { ...b, value } : b)),
      );
      setEditedBlocks((prev) => {
        const next = { ...prev };
        delete next[label];
        return next;
      });
    } catch {
      // handle silently
    } finally {
      setSaving(null);
    }
  }

  async function handleReset() {
    if (!confirm('Reset all memory? This cannot be undone.')) return;
    try {
      await resetMemory();
      await load();
    } catch {
      // handle silently
    }
  }

  async function handleSearch() {
    if (!searchQuery.trim()) return;
    setSearching(true);
    try {
      const results = await searchKnowledgeEntries({
        query: searchQuery.trim(),
        k: 20,
      });
      setKnowledge(results);
    } catch {
      // handle silently
    } finally {
      setSearching(false);
    }
  }

  async function handleAddEntry() {
    if (!newEntry.trim()) return;
    setAdding(true);
    try {
      const entry = await createKnowledgeEntry({
        content: newEntry.trim(),
        source: 'user',
      });
      setKnowledge((prev) => [entry, ...prev]);
      setNewEntry('');
    } catch {
      // handle silently
    } finally {
      setAdding(false);
    }
  }

  async function handleDeleteEntry(id: string) {
    try {
      await deleteKnowledgeEntry(id);
      setKnowledge((prev) => prev.filter((e) => e.id !== id));
    } catch {
      // handle silently
    }
  }

  if (loading) {
    return (
      <div className="flex justify-center py-20">
        <Spinner className="h-8 w-8" />
      </div>
    );
  }

  return (
    <div className="p-6 lg:p-8">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-zinc-100">Memory</h1>
          <p className="mt-1 text-sm text-zinc-400">
            Persistent agent memory shared across conversations
          </p>
        </div>
        <Button variant="danger" size="sm" onClick={handleReset}>
          <RotateCcw size={14} /> Reset All
        </Button>
      </div>

      {/* Core Memory Blocks */}
      <div className="mb-8 space-y-4">
        <h2 className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wider text-zinc-400">
          <Brain size={16} /> Core Memory
        </h2>
        <div className="grid gap-4 lg:grid-cols-3">
          {blocks.map((block) => {
            const edited = editedBlocks[block.label];
            const isDirty = edited !== undefined && edited !== block.value;

            return (
              <Card key={block.label} className="space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold capitalize text-zinc-200">
                    {block.label}
                  </h3>
                  <span className="text-xs text-zinc-500">
                    {(edited ?? block.value).length}/{block.limit}
                  </span>
                </div>
                <TextArea
                  rows={5}
                  value={edited ?? block.value}
                  onChange={(e) =>
                    setEditedBlocks((prev) => ({
                      ...prev,
                      [block.label]: e.target.value,
                    }))
                  }
                  disabled={block.read_only}
                />
                {isDirty && (
                  <Button
                    size="sm"
                    onClick={() => saveBlock(block.label)}
                    disabled={saving === block.label}
                  >
                    {saving === block.label ? (
                      <Spinner className="h-3 w-3" />
                    ) : (
                      <Save size={14} />
                    )}
                    Save
                  </Button>
                )}
              </Card>
            );
          })}
        </div>
      </div>

      {/* Knowledge Entries */}
      <div className="space-y-4">
        <h2 className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wider text-zinc-400">
          <Search size={16} /> Knowledge
        </h2>

        <div className="flex gap-3">
          <Input
            placeholder="Add a long-term fact..."
            value={newEntry}
            onChange={(e) => setNewEntry(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleAddEntry()}
            className="flex-1"
          />
          <Button onClick={handleAddEntry} disabled={adding || !newEntry.trim()}>
            {adding ? <Spinner className="h-4 w-4" /> : <Plus size={16} />}
          </Button>
        </div>

        <div className="flex gap-3">
          <Input
            placeholder="Search knowledge..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
            className="flex-1"
          />
          <Button onClick={handleSearch} disabled={searching}>
            {searching ? <Spinner className="h-4 w-4" /> : <Search size={16} />}
          </Button>
          <Button variant="secondary" onClick={() => listKnowledgeEntries().then(setKnowledge)}>
            Show All
          </Button>
        </div>

        {knowledge.length === 0 ? (
          <Card className="py-8 text-center text-sm text-zinc-500">
            No knowledge entries found
          </Card>
        ) : (
          <div className="space-y-3">
            {knowledge.map((entry) => (
              <Card key={entry.id} className="space-y-2">
                <p className="text-sm text-zinc-300">{entry.content}</p>
                <div className="flex items-center justify-between">
                  <div className="flex flex-wrap gap-2">
                    {entry.tags?.map((tag) => (
                      <Badge key={tag} color="blue">
                        {tag}
                      </Badge>
                    ))}
                    <Badge color="gray">importance: {entry.importance}</Badge>
                    {entry.kind && <Badge color="accent">{entry.kind}</Badge>}
                    <Badge color="gray">{entry.source}</Badge>
                  </div>
                  <button onClick={() => handleDeleteEntry(entry.id)}>
                    <Trash2
                      size={14}
                      className="text-zinc-500 hover:text-red-400"
                    />
                  </button>
                </div>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
