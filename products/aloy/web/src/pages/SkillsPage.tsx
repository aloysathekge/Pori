import { useCallback, useEffect, useState, type ReactNode } from 'react';
import { Plus, Save, X } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Spinner } from '@/components/ui/Spinner';
import {
  createSkill,
  listSkills,
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
  'w-full rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm text-zinc-100 placeholder-zinc-500 focus:border-indigo-500 focus:outline-none';

export function SkillsPage() {
  const [skills, setSkills] = useState<SkillResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [editing, setEditing] = useState<SkillResponse | null>(null);
  const [creating, setCreating] = useState(false);
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
    load();
  }, [load]);

  const set = (patch: Partial<SkillCreate>) =>
    setForm((f) => ({ ...f, ...patch }));

  function startCreate() {
    setEditing(null);
    setForm(EMPTY);
    setError('');
    setCreating(true);
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
      <div className="flex items-center justify-between border-b border-zinc-800 px-6 py-4">
        <div>
          <h1 className="text-lg font-semibold text-zinc-100">Skills</h1>
          <p className="text-sm text-zinc-400">
            Reusable instructions the agent loads on demand.
          </p>
        </div>
        {!creating && (
          <Button onClick={startCreate} className="gap-2">
            <Plus size={16} /> New skill
          </Button>
        )}
      </div>

      <div className="flex-1 overflow-y-auto p-6">
        {error && (
          <div className="mb-4 rounded-lg border border-red-800 bg-red-950/50 px-4 py-2 text-sm text-red-300">
            {error}
          </div>
        )}

        {creating ? (
          <div className="mx-auto max-w-2xl space-y-4">
            <h2 className="text-base font-medium text-zinc-200">
              {editing ? `Edit ${editing.name}` : 'New skill'}
            </h2>
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
            <div className="grid grid-cols-2 gap-4">
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
