import { useCallback, useEffect, useState, type ReactNode } from 'react';
import { Pause, Play, Plus, Save, Trash2, X } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Spinner } from '@/components/ui/Spinner';
import {
  createSchedule,
  deleteSchedule,
  listSchedules,
  updateSchedule,
  type ScheduleCreate,
  type ScheduleResponse,
} from '@/api/schedules';

const EMPTY: ScheduleCreate = {
  name: '',
  task: '',
  schedule: '@every:86400',
  max_steps: 15,
};

const INPUT =
  'w-full rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm text-zinc-100 placeholder-zinc-500 focus:border-indigo-500 focus:outline-none';

function formatWhen(iso: string | null): string {
  if (!iso) return '—';
  const date = new Date(iso.endsWith('Z') || iso.includes('+') ? iso : `${iso}Z`);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString();
}

export function SchedulesPage() {
  const [schedules, setSchedules] = useState<ScheduleResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [form, setForm] = useState<ScheduleCreate>(EMPTY);
  const [saving, setSaving] = useState(false);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [error, setError] = useState('');

  // `loading` starts true and is only cleared — reloads after a mutation keep
  // the current list on screen instead of flashing the spinner (and keeping
  // setState out of the effect's synchronous path).
  const load = useCallback(async () => {
    try {
      setSchedules(await listSchedules());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    // Fetch-on-mount: all setState calls happen after the await, so nothing
    // updates synchronously inside the effect body.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    load();
  }, [load]);

  const set = (patch: Partial<ScheduleCreate>) =>
    setForm((f) => ({ ...f, ...patch }));

  async function save() {
    setSaving(true);
    setError('');
    try {
      await createSchedule(form);
      setCreating(false);
      setForm(EMPTY);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }

  async function toggle(schedule: ScheduleResponse) {
    setBusyId(schedule.id);
    setError('');
    try {
      await updateSchedule(schedule.id, { enabled: !schedule.enabled });
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusyId(null);
    }
  }

  async function remove(schedule: ScheduleResponse) {
    if (!window.confirm(`Delete schedule "${schedule.name}"?`)) return;
    setBusyId(schedule.id);
    setError('');
    try {
      await deleteSchedule(schedule.id);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusyId(null);
    }
  }

  const canSave = !!form.name && !!form.task && !!form.schedule;

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-zinc-800 px-6 py-4">
        <div>
          <h1 className="text-lg font-semibold text-zinc-100">Schedules</h1>
          <p className="text-sm text-zinc-400">
            Recurring tasks your agent runs on its own — durable, resumable,
            delivered when done.
          </p>
        </div>
        {!creating && (
          <Button onClick={() => setCreating(true)} className="gap-2">
            <Plus size={16} /> New schedule
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
            <h2 className="text-base font-medium text-zinc-200">New schedule</h2>
            <Field label="Name">
              <input
                className={INPUT}
                value={form.name}
                onChange={(e) => set({ name: e.target.value })}
                placeholder="Morning briefing"
              />
            </Field>
            <Field label="Task — what should the agent do each time?">
              <textarea
                className={`${INPUT} min-h-28`}
                value={form.task}
                onChange={(e) => set({ task: e.target.value })}
                placeholder="Summarize my unread items and flag anything urgent."
              />
            </Field>
            <div className="grid grid-cols-2 gap-4">
              <Field label="When — cron ('0 7 * * 1-5') or '@every:SECONDS'">
                <input
                  className={INPUT}
                  value={form.schedule}
                  onChange={(e) => set({ schedule: e.target.value })}
                  placeholder="0 7 * * 1-5"
                />
              </Field>
              <Field label="Max steps per run">
                <input
                  className={INPUT}
                  type="number"
                  min={1}
                  value={form.max_steps ?? 15}
                  onChange={(e) =>
                    set({ max_steps: Number(e.target.value) || 15 })
                  }
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
                  setError('');
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
        ) : schedules.length === 0 ? (
          <p className="py-12 text-center text-sm text-zinc-500">
            No schedules yet. Create one and your agent will run it on time,
            every time — even across restarts.
          </p>
        ) : (
          <div className="mx-auto grid max-w-3xl gap-3">
            {schedules.map((s) => (
              <div
                key={s.id}
                className="rounded-xl border border-zinc-800 bg-zinc-900 p-4"
              >
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-zinc-100">{s.name}</span>
                    <span
                      className={`rounded-full px-2 py-0.5 text-xs ${
                        s.enabled
                          ? 'bg-emerald-950 text-emerald-400'
                          : 'bg-zinc-800 text-zinc-500'
                      }`}
                    >
                      {s.enabled ? 'active' : 'paused'}
                    </span>
                  </div>
                  <div className="flex items-center gap-1">
                    <button
                      type="button"
                      aria-label={s.enabled ? 'Pause schedule' : 'Resume schedule'}
                      disabled={busyId === s.id}
                      onClick={() => toggle(s)}
                      className="rounded-lg p-2 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-50"
                    >
                      {s.enabled ? <Pause size={16} /> : <Play size={16} />}
                    </button>
                    <button
                      type="button"
                      aria-label="Delete schedule"
                      disabled={busyId === s.id}
                      onClick={() => remove(s)}
                      className="rounded-lg p-2 text-zinc-400 hover:bg-zinc-800 hover:text-red-400 disabled:opacity-50"
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                </div>
                <p className="mt-1 line-clamp-2 text-sm text-zinc-400">{s.task}</p>
                <div className="mt-3 flex flex-wrap gap-x-6 gap-y-1 text-xs text-zinc-500">
                  <span>
                    <span className="text-zinc-600">runs</span>{' '}
                    <code className="text-zinc-400">{s.schedule}</code>
                  </span>
                  <span>
                    <span className="text-zinc-600">next</span>{' '}
                    {s.enabled ? formatWhen(s.next_run_at) : '—'}
                  </span>
                  <span>
                    <span className="text-zinc-600">last</span>{' '}
                    {formatWhen(s.last_run_at)}
                  </span>
                </div>
              </div>
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
