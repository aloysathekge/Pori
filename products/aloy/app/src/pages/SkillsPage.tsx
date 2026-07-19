import { useCallback, useEffect, useState, type ReactNode } from 'react';
import { Download, Loader2, Plus, Save, Upload, X } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Spinner } from '@/components/ui/Spinner';
import {
  createSkill,
  listSkills,
  previewSkillImport,
  updateSkill,
  type SkillCreate,
  type SkillResponse,
} from '@/api/skills';

const EMPTY: SkillCreate = {
  slug: '',
  version: '1',
  name: '',
  summary: '',
  instructions: '',
  tags: [],
  category: 'organization',
};

const INPUT =
  'w-full rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm text-zinc-100 placeholder-zinc-500 focus:border-accent-500 focus:outline-none';

export function SkillsPage() {
  const [skills, setSkills] = useState<SkillResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [editing, setEditing] = useState<SkillResponse | null>(null);
  const [creating, setCreating] = useState(false);
  const [importing, setImporting] = useState(false);
  const [importUrl, setImportUrl] = useState('');
  const [importText, setImportText] = useState('');
  const [importBusy, setImportBusy] = useState(false);
  const [importWarnings, setImportWarnings] = useState<string[]>([]);
  const [form, setForm] = useState<SkillCreate>(EMPTY);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setSkills(await listSkills());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    // Mount-time fetch; `loading` already initializes true (no cascade).
    // eslint-disable-next-line react-hooks/set-state-in-effect
    load();
  }, [load]);

  const set = (patch: Partial<SkillCreate>) =>
    setForm((f) => ({ ...f, ...patch }));

  function startCreate() {
    setEditing(null);
    setForm(EMPTY);
    setError('');
    setImporting(false);
    setCreating(true);
  }

  function startImport() {
    setEditing(null);
    setError('');
    setImportUrl('');
    setImportText('');
    setImportWarnings([]);
    setCreating(false);
    setImporting(true);
  }

  /** Parse the pasted URL/text server-side, then drop the user into the
   *  normal editor with everything prefilled — review, tweak, save. */
  async function runImportPreview(input: { url?: string; text?: string }) {
    setImportBusy(true);
    setError('');
    try {
      const p = await previewSkillImport(input);
      setForm({
        slug: p.slug,
        version: p.version,
        name: p.name,
        summary: p.summary,
        instructions: p.instructions,
        tags: p.tags,
        category: p.category,
      });
      setImportWarnings(p.warnings);
      setImporting(false);
      setCreating(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setImportBusy(false);
    }
  }

  function onImportFile(file: File) {
    const reader = new FileReader();
    reader.onload = () => {
      const text = String(reader.result || '');
      if (text) void runImportPreview({ text });
    };
    reader.readAsText(file);
  }

  function startEdit(skill: SkillResponse) {
    setEditing(skill);
    setError('');
    setForm({
      slug: skill.slug,
      version: skill.version,
      name: skill.name,
      summary: skill.summary,
      instructions: skill.instructions,
      tags: skill.tags,
      category: skill.category,
    });
    setCreating(true);
  }

  async function save() {
    setSaving(true);
    setError('');
    try {
      if (editing) {
        const { slug: _slug, ...rest } = form;
        await updateSkill(editing.id, rest);
      } else {
        await createSkill(form);
      }
      setCreating(false);
      setEditing(null);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }

  const canSave =
    !!form.name && !!form.summary && !!form.instructions && (!!editing || !!form.slug);

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-zinc-800 px-4 py-3 sm:px-6 sm:py-4">
        <div>
          <h1 className="text-lg font-semibold text-zinc-100">Skills</h1>
          <p className="text-sm text-zinc-400">
            Reusable instructions the agent loads on demand.
          </p>
        </div>
        {!creating && !importing && (
          <div className="flex gap-2">
            <Button onClick={startImport} className="gap-2">
              <Download size={16} /> Import
            </Button>
            <button
              type="button"
              onClick={startCreate}
              className="inline-flex items-center gap-2 rounded-lg border border-zinc-700 px-3 py-2 text-sm text-zinc-300 hover:border-zinc-600"
            >
              <Plus size={16} /> Write from scratch
            </button>
          </div>
        )}
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-5 sm:p-6">
        {error && (
          <div className="mb-4 rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-2 text-sm text-red-600">
            {error}
          </div>
        )}

        {importing ? (
          <div className="mx-auto max-w-2xl space-y-4">
            <h2 className="text-base font-medium text-zinc-200">Import a skill</h2>
            <p className="text-sm text-zinc-400">
              Skills use the standard <code className="text-zinc-300">SKILL.md</code>{' '}
              format (YAML frontmatter + markdown instructions). Paste a link —
              GitHub links work directly — or the file itself.
            </p>
            <Field label="From a URL">
              <div className="flex gap-2">
                <input
                  className={INPUT}
                  value={importUrl}
                  onChange={(e) => setImportUrl(e.target.value)}
                  placeholder="https://github.com/…/SKILL.md"
                />
                <Button
                  onClick={() => void runImportPreview({ url: importUrl.trim() })}
                  disabled={importBusy || !importUrl.trim()}
                  className="shrink-0 gap-2"
                >
                  {importBusy ? <Loader2 size={16} className="animate-spin" /> : <Download size={16} />}
                  Fetch
                </Button>
              </div>
            </Field>
            <div className="text-center text-xs text-zinc-600">— or —</div>
            <Field label="Paste the SKILL.md contents">
              <textarea
                className={`${INPUT} min-h-40 font-mono`}
                value={importText}
                onChange={(e) => setImportText(e.target.value)}
                placeholder={'---\nname: My skill\ndescription: …\n---\nInstructions…'}
              />
            </Field>
            <div className="flex gap-2">
              <Button
                onClick={() => void runImportPreview({ text: importText })}
                disabled={importBusy || !importText.trim()}
                className="gap-2"
              >
                {importBusy ? <Loader2 size={16} className="animate-spin" /> : <Upload size={16} />}
                Preview
              </Button>
              <label className="inline-flex cursor-pointer items-center gap-2 rounded-lg border border-zinc-700 px-3 py-2 text-sm text-zinc-300 hover:border-zinc-600">
                <Upload size={16} /> Upload .md file
                <input
                  type="file"
                  accept=".md,text/markdown,text/plain"
                  className="hidden"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) onImportFile(f);
                    e.target.value = '';
                  }}
                />
              </label>
              <button
                type="button"
                onClick={() => setImporting(false)}
                className="inline-flex items-center gap-2 rounded-lg border border-zinc-700 px-3 py-2 text-sm text-zinc-300 hover:border-zinc-600"
              >
                <X size={16} /> Cancel
              </button>
            </div>
          </div>
        ) : creating ? (
          <div className="mx-auto max-w-2xl space-y-4">
            <h2 className="text-base font-medium text-zinc-200">
              {editing ? `Edit ${editing.name}` : 'New skill'}
            </h2>
            {importWarnings.length > 0 && (
              <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-4 py-2 text-sm text-amber-300">
                {importWarnings.map((w) => (
                  <p key={w}>{w}</p>
                ))}
              </div>
            )}
            {!editing && (
              <Field label="Slug">
                <input
                  className={INPUT}
                  value={form.slug}
                  onChange={(e) => set({ slug: e.target.value })}
                  placeholder="my-skill"
                />
              </Field>
            )}
            <Field label="Name">
              <input
                className={INPUT}
                value={form.name}
                onChange={(e) => set({ name: e.target.value })}
              />
            </Field>
            <Field label="Summary">
              <input
                className={INPUT}
                value={form.summary}
                onChange={(e) => set({ summary: e.target.value })}
              />
            </Field>
            <Field label="Instructions">
              <textarea
                className={`${INPUT} min-h-40 font-mono`}
                value={form.instructions}
                onChange={(e) => set({ instructions: e.target.value })}
              />
            </Field>
            <div className="grid gap-4 sm:grid-cols-2">
              <Field label="Tags (comma-separated)">
                <input
                  className={INPUT}
                  value={(form.tags ?? []).join(', ')}
                  onChange={(e) =>
                    set({
                      tags: e.target.value
                        .split(',')
                        .map((t) => t.trim())
                        .filter(Boolean),
                    })
                  }
                />
              </Field>
              <Field label="Category">
                <input
                  className={INPUT}
                  value={form.category}
                  onChange={(e) => set({ category: e.target.value })}
                />
              </Field>
            </div>
            <div className="flex gap-2">
              <Button onClick={save} disabled={saving || !canSave} className="gap-2">
                <Save size={16} /> {saving ? 'Saving…' : 'Save'}
              </Button>
              <button
                type="button"
                onClick={() => {
                  setCreating(false);
                  setEditing(null);
                }}
                className="inline-flex items-center gap-2 rounded-lg border border-zinc-700 px-3 py-2 text-sm text-zinc-300 hover:border-zinc-600"
              >
                <X size={16} /> Cancel
              </button>
            </div>
          </div>
        ) : loading ? (
          <div className="flex justify-center py-12">
            <Spinner className="h-8 w-8" />
          </div>
        ) : skills.length === 0 ? (
          <p className="py-12 text-center text-sm text-zinc-500">
            No skills yet. Create one to get started.
          </p>
        ) : (
          <div className="mx-auto grid max-w-3xl gap-3">
            {skills.map((s) => (
              <button
                key={s.id}
                type="button"
                onClick={() => startEdit(s)}
                className="rounded-xl border border-zinc-800 bg-zinc-900 p-4 text-left hover:border-zinc-700"
              >
                <div className="flex items-center justify-between">
                  <span className="font-medium text-zinc-100">{s.name}</span>
                  <span className="text-xs text-zinc-500">
                    {s.category} · v{s.version}
                  </span>
                </div>
                <p className="mt-1 text-sm text-zinc-400">{s.summary}</p>
                {s.tags.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {s.tags.map((t) => (
                      <span
                        key={t}
                        className="rounded-full bg-zinc-800 px-2 py-0.5 text-xs text-zinc-400"
                      >
                        {t}
                      </span>
                    ))}
                  </div>
                )}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs font-medium text-zinc-400">{label}</span>
      {children}
    </label>
  );
}
