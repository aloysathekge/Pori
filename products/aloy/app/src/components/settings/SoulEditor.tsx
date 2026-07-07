import { useEffect, useRef, useState } from 'react';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { TextArea } from '@/components/ui/TextArea';
import { Spinner } from '@/components/ui/Spinner';
import { getSoul, setSoul, clearSoul } from '@/api/soul';

export function SoulEditor() {
  const [content, setContent] = useState('');
  const [limit, setLimit] = useState(8000);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState('');
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    getSoul()
      .then((s) => {
        setContent(s.content);
        setLimit(s.char_limit);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  async function handleSave() {
    setSaving(true);
    setError('');
    setSaved(false);
    try {
      const s = await setSoul(content);
      setContent(s.content);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save SOUL');
    } finally {
      setSaving(false);
    }
  }

  async function handleClear() {
    setSaving(true);
    setError('');
    try {
      await clearSoul();
      setContent('');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to clear SOUL');
    } finally {
      setSaving(false);
    }
  }

  function handleImport(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () =>
      setContent(String(reader.result || '').slice(0, limit));
    reader.readAsText(file);
    e.target.value = '';
  }

  if (loading) {
    return (
      <Card className="flex justify-center py-8">
        <Spinner className="h-5 w-5" />
      </Card>
    );
  }

  return (
    <Card className="space-y-3">
      <div>
        <h2 className="text-sm font-semibold text-zinc-300">Identity (SOUL)</h2>
        <p className="mt-1 text-xs text-zinc-500">
          Your agent's persona, applied to every run. Write it here or import a
          SOUL.md file. Leave it empty to use the default identity.
        </p>
      </div>

      <TextArea
        value={content}
        onChange={(e) => setContent(e.target.value.slice(0, limit))}
        rows={8}
        placeholder="You are Nova, a blunt and witty code reviewer who values correctness over politeness…"
      />

      <div className="flex flex-wrap items-center gap-3">
        <Button onClick={handleSave} disabled={saving}>
          {saving ? <Spinner className="h-4 w-4" /> : 'Save'}
        </Button>
        <Button
          variant="outline"
          onClick={() => fileRef.current?.click()}
          disabled={saving}
        >
          Import .md
        </Button>
        <Button
          variant="ghost"
          onClick={handleClear}
          disabled={saving || !content}
        >
          Clear
        </Button>
        <input
          ref={fileRef}
          type="file"
          accept=".md,.txt,text/markdown,text/plain"
          className="hidden"
          onChange={handleImport}
        />
        <span className="ml-auto text-xs text-zinc-500">
          {content.length}/{limit}
        </span>
        {saved && <span className="text-xs text-emerald-700">Saved!</span>}
      </div>

      {error && <p className="text-xs text-red-600">{error}</p>}
    </Card>
  );
}
