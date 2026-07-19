import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react';
import {
  AlertCircle,
  CalendarClock,
  ChevronDown,
  ChevronUp,
  Clock3,
  Edit3,
  History,
  Pause,
  Play,
  Plus,
  Save,
  ShieldCheck,
  Trash2,
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { listEvents, type EventSummary } from '@/api/events';
import {
  createSchedule,
  deleteSchedule,
  listScheduleRuns,
  listSchedules,
  updateSchedule,
  type ScheduleCreate,
  type ScheduleResponse,
  type ScheduleRunResponse,
} from '@/api/schedules';
import { EventIcon } from '@/components/icons';
import { Button } from '@/components/ui/Button';
import { Modal } from '@/components/ui/Modal';
import { Spinner } from '@/components/ui/Spinner';
import { formatRelativeTime } from '@/lib/time';

type Cadence = 'daily' | 'weekdays' | 'weekly' | 'advanced';

interface ScheduleForm {
  event_id: string;
  name: string;
  task: string;
  cadence: Cadence;
  time: string;
  weekday: string;
  advanced: string;
  timezone: string;
  authority: ScheduleCreate['authority'];
  notification_mode: ScheduleCreate['notification_mode'];
  max_steps: number;
}

interface RunHistoryState {
  loading: boolean;
  runs: ScheduleRunResponse[];
  error?: string;
}

const INPUT =
  'w-full rounded-xl border border-zinc-700 bg-zinc-900 px-3.5 py-2.5 text-sm text-zinc-100 placeholder-zinc-600 outline-none transition focus:border-accent-600 focus:ring-2 focus:ring-accent-900/30';

const WEEKDAYS = [
  ['1', 'Monday'],
  ['2', 'Tuesday'],
  ['3', 'Wednesday'],
  ['4', 'Thursday'],
  ['5', 'Friday'],
  ['6', 'Saturday'],
  ['0', 'Sunday'],
] as const;

function detectedTimezone() {
  return Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC';
}

function emptyForm(eventId = ''): ScheduleForm {
  return {
    event_id: eventId,
    name: '',
    task: '',
    cadence: 'weekdays',
    time: '08:00',
    weekday: '1',
    advanced: '0 8 * * 1-5',
    timezone: detectedTimezone(),
    authority: 'report_only',
    notification_mode: 'attention',
    max_steps: 15,
  };
}

function expressionFrom(form: ScheduleForm) {
  if (form.cadence === 'advanced') return form.advanced.trim();
  const [hour = '8', minute = '0'] = form.time.split(':');
  const day = form.cadence === 'daily' ? '*' : form.cadence === 'weekdays' ? '1-5' : form.weekday;
  return `${Number(minute)} ${Number(hour)} * * ${day}`;
}

function parseTiming(schedule: ScheduleResponse): Pick<ScheduleForm, 'cadence' | 'time' | 'weekday' | 'advanced'> {
  const fields = schedule.schedule.trim().split(/\s+/);
  if (fields.length === 5 && fields[2] === '*' && fields[3] === '*') {
    const [minute, hour, , , day] = fields;
    const time = `${String(Number(hour)).padStart(2, '0')}:${String(Number(minute)).padStart(2, '0')}`;
    if (day === '*') return { cadence: 'daily', time, weekday: '1', advanced: schedule.schedule };
    if (day === '1-5') return { cadence: 'weekdays', time, weekday: '1', advanced: schedule.schedule };
    if (WEEKDAYS.some(([value]) => value === day)) {
      return { cadence: 'weekly', time, weekday: day!, advanced: schedule.schedule };
    }
  }
  return { cadence: 'advanced', time: '08:00', weekday: '1', advanced: schedule.schedule };
}

function editForm(schedule: ScheduleResponse): ScheduleForm {
  return {
    event_id: schedule.event_id ?? '',
    name: schedule.name,
    task: schedule.task,
    timezone: schedule.timezone,
    authority: schedule.authority,
    notification_mode: schedule.notification_mode,
    max_steps: schedule.max_steps,
    ...parseTiming(schedule),
  };
}

function scheduleLabel(schedule: ScheduleResponse) {
  const parsed = parseTiming(schedule);
  if (parsed.cadence === 'advanced') return schedule.schedule.startsWith('@every:') ? 'Repeating interval' : 'Custom schedule';
  const time = parsed.time;
  if (parsed.cadence === 'daily') return `Every day at ${time}`;
  if (parsed.cadence === 'weekdays') return `Weekdays at ${time}`;
  const weekday = WEEKDAYS.find(([value]) => value === parsed.weekday)?.[1] ?? 'week';
  return `Every ${weekday} at ${time}`;
}

function formatWhen(iso: string | null, timezone: string) {
  if (!iso) return 'Not scheduled';
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  try {
    return new Intl.DateTimeFormat(undefined, {
      weekday: 'short',
      day: 'numeric',
      month: 'short',
      hour: '2-digit',
      minute: '2-digit',
      timeZone: timezone,
      timeZoneName: 'short',
    }).format(date);
  } catch {
    return date.toLocaleString();
  }
}

function runStatus(run: ScheduleRunResponse) {
  if (run.status === 'running') return 'In progress';
  if (run.status === 'pending') return 'Queued';
  if (run.status === 'completed' && run.success) return 'Completed';
  if (run.status === 'completed') return 'Needs attention';
  return run.status.charAt(0).toUpperCase() + run.status.slice(1);
}

export function SchedulesPage() {
  const navigate = useNavigate();
  const [schedules, setSchedules] = useState<ScheduleResponse[]>([]);
  const [events, setEvents] = useState<EventSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [editorOpen, setEditorOpen] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [form, setForm] = useState<ScheduleForm>(() => emptyForm());
  const [saving, setSaving] = useState(false);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [histories, setHistories] = useState<Record<string, RunHistoryState>>({});
  const [error, setError] = useState('');

  const dedicatedEvents = useMemo(
    () => events.filter((event) => !event.is_life && event.lifecycle === 'active'),
    [events],
  );
  const eventById = useMemo(
    () => new Map(dedicatedEvents.map((event) => [event.id, event])),
    [dedicatedEvents],
  );

  const load = useCallback(async () => {
    try {
      const [loadedSchedules, loadedEvents] = await Promise.all([listSchedules(), listEvents()]);
      setSchedules(loadedSchedules);
      setEvents(loadedEvents);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  function openCreate() {
    setEditingId(null);
    setForm(emptyForm(dedicatedEvents[0]?.id ?? ''));
    setError('');
    setEditorOpen(true);
  }

  function openEdit(schedule: ScheduleResponse) {
    setEditingId(schedule.id);
    setForm(editForm(schedule));
    setError('');
    setEditorOpen(true);
  }

  function set(patch: Partial<ScheduleForm>) {
    setForm((current) => ({ ...current, ...patch }));
  }

  async function save() {
    setSaving(true);
    setError('');
    const payload = {
      name: form.name.trim(),
      task: form.task.trim(),
      schedule: expressionFrom(form),
      timezone: form.timezone.trim(),
      authority: form.authority,
      notification_mode: form.notification_mode,
      max_steps: form.max_steps,
    };
    try {
      if (editingId) {
        await updateSchedule(editingId, payload);
      } else {
        await createSchedule({ ...payload, event_id: form.event_id });
      }
      setEditorOpen(false);
      setEditingId(null);
      await load();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
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
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusyId(null);
    }
  }

  async function remove(schedule: ScheduleResponse) {
    if (!window.confirm(`Delete schedule "${schedule.name}"? Its Trail and run receipts will remain.`)) return;
    setBusyId(schedule.id);
    setError('');
    try {
      await deleteSchedule(schedule.id);
      await load();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusyId(null);
    }
  }

  async function toggleHistory(schedule: ScheduleResponse) {
    if (expandedId === schedule.id) {
      setExpandedId(null);
      return;
    }
    setExpandedId(schedule.id);
    if (histories[schedule.id]) return;
    setHistories((current) => ({ ...current, [schedule.id]: { loading: true, runs: [] } }));
    try {
      const runs = await listScheduleRuns(schedule.id);
      setHistories((current) => ({ ...current, [schedule.id]: { loading: false, runs } }));
    } catch (cause) {
      setHistories((current) => ({
        ...current,
        [schedule.id]: {
          loading: false,
          runs: [],
          error: cause instanceof Error ? cause.message : String(cause),
        },
      }));
    }
  }

  const canSave = Boolean(
    form.event_id && form.name.trim() && form.task.trim() && expressionFrom(form) && form.timezone.trim(),
  );

  return (
    <div className="h-full overflow-y-auto bg-zinc-950">
      <div className="mx-auto min-h-full max-w-[1180px] px-5 py-7 lg:px-10 lg:py-9">
        <header className="flex flex-col gap-5 border-b border-zinc-800 pb-7 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="flex items-center gap-2 text-sm font-medium text-accent-700">
              <CalendarClock size={16} /> Event automations
            </p>
            <h1 className="mt-2 font-display text-3xl font-semibold tracking-tight text-zinc-100">Schedules</h1>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-zinc-400">
              Give Aloy a durable reason to wake inside an Event. Every run keeps its authority, history, and Trail evidence.
            </p>
          </div>
          <Button onClick={openCreate} disabled={dedicatedEvents.length === 0}>
            <Plus size={16} /> New schedule
          </Button>
        </header>

        {error && !editorOpen && (
          <div className="mt-5 flex items-start gap-3 rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-300">
            <AlertCircle size={17} className="mt-0.5 shrink-0" /> {error}
          </div>
        )}

        {loading ? (
          <div className="flex h-72 items-center justify-center"><Spinner /></div>
        ) : dedicatedEvents.length === 0 ? (
          <div className="mt-10 border-y border-zinc-800 py-12 text-center">
            <EventIcon size={28} className="mx-auto text-zinc-600" />
            <h2 className="mt-4 font-display text-xl font-semibold text-zinc-200">Create an Event first</h2>
            <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-zinc-500">Schedules belong to one dedicated Event so context, permissions, results, and memory never drift into the wrong part of your life.</p>
            <Button className="mt-5" onClick={() => navigate('/events/new')}>Create an Event</Button>
          </div>
        ) : schedules.length === 0 ? (
          <div className="mt-10 grid gap-8 border-y border-zinc-800 py-10 md:grid-cols-[1fr_0.85fr] md:items-center">
            <div>
              <h2 className="font-display text-2xl font-semibold text-zinc-100">What should Aloy keep up with?</h2>
              <p className="mt-3 max-w-xl text-sm leading-6 text-zinc-400">A morning university check, a weekly job pipeline review, or a deadline watch can run without you reopening the Event.</p>
              <Button className="mt-5" onClick={openCreate}><Plus size={16} /> Create your first schedule</Button>
            </div>
            <div className="space-y-3 border-l border-zinc-800 pl-0 md:pl-8">
              <TrustLine>Runs only inside the Event you choose.</TrustLine>
              <TrustLine>External actions still require approval.</TrustLine>
              <TrustLine>Every wake, outcome, and failure stays in the Trail.</TrustLine>
            </div>
          </div>
        ) : (
          <main className="mt-7 space-y-4">
            {schedules.map((schedule) => {
              const event = schedule.event_id ? eventById.get(schedule.event_id) : null;
              const history = histories[schedule.id];
              const expanded = expandedId === schedule.id;
              return (
                <article key={schedule.id} className="overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-900/35">
                  <div className="p-5 sm:p-6">
                    <div className="flex flex-col gap-5 lg:flex-row lg:items-start lg:justify-between">
                      <div className="min-w-0 flex-1">
                        <div className="flex flex-wrap items-center gap-2">
                          <h2 className="font-display text-xl font-semibold text-zinc-100">{schedule.name}</h2>
                          <span className={`rounded-full px-2.5 py-1 text-xs font-medium ${schedule.enabled ? 'bg-emerald-500/12 text-emerald-400' : 'bg-zinc-800 text-zinc-500'}`}>{schedule.enabled ? 'Active' : 'Paused'}</span>
                        </div>
                        <button type="button" disabled={!event} onClick={() => event && navigate(`/events/${event.id}`)} className="mt-2 inline-flex items-center gap-2 text-sm font-medium text-accent-700 hover:text-accent-600 disabled:text-zinc-600">
                          <EventIcon size={14} /> {event?.title ?? 'Legacy schedule without an Event'}
                        </button>
                        <p className="mt-4 max-w-3xl text-sm leading-6 text-zinc-300">{schedule.task}</p>
                      </div>
                      <div className="flex shrink-0 items-center gap-1">
                        <IconButton label="Edit schedule" disabled={!event} onClick={() => openEdit(schedule)}><Edit3 size={16} /></IconButton>
                        <IconButton label={schedule.enabled ? 'Pause schedule' : 'Resume schedule'} disabled={busyId === schedule.id || !event} onClick={() => void toggle(schedule)}>{schedule.enabled ? <Pause size={16} /> : <Play size={16} />}</IconButton>
                        <IconButton label="Delete schedule" disabled={busyId === schedule.id} onClick={() => void remove(schedule)} danger><Trash2 size={16} /></IconButton>
                      </div>
                    </div>

                    <div className="mt-5 grid gap-4 border-t border-zinc-800 pt-5 sm:grid-cols-2 lg:grid-cols-4">
                      <Fact label="Timing" value={scheduleLabel(schedule)} detail={schedule.timezone} icon={<CalendarClock size={16} />} />
                      <Fact label="Next wake" value={schedule.enabled ? formatWhen(schedule.next_run_at, schedule.timezone) : 'Paused'} detail={schedule.enabled ? 'The worker will claim it durably' : 'No work will be queued'} icon={<Clock3 size={16} />} />
                      <Fact label="Authority" value={schedule.authority === 'report_only' ? 'Read and report' : 'Organize this Event'} detail="Protected external actions require approval" icon={<ShieldCheck size={16} />} />
                      <Fact label="Notify" value={schedule.notification_mode === 'always' ? 'After every run' : 'Only when attention is needed'} detail={`Up to ${schedule.max_steps} steps per run`} icon={<AlertCircle size={16} />} />
                    </div>
                  </div>

                  <button type="button" onClick={() => void toggleHistory(schedule)} className="flex w-full items-center justify-between border-t border-zinc-800 bg-zinc-900/60 px-5 py-3 text-left text-sm font-medium text-zinc-400 hover:text-zinc-200 sm:px-6">
                    <span className="flex items-center gap-2"><History size={15} /> Run history{schedule.last_run_at ? ` · last ${formatRelativeTime(schedule.last_run_at)}` : ''}</span>
                    {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                  </button>
                  {expanded && (
                    <RunHistory history={history} timezone={schedule.timezone} />
                  )}
                </article>
              );
            })}
          </main>
        )}
      </div>

      <Modal
        open={editorOpen}
        onClose={() => !saving && setEditorOpen(false)}
        title={editingId ? 'Edit Event Schedule' : 'New Event Schedule'}
        panelClassName="max-w-3xl"
      >
        <div className="max-h-[76vh] overflow-y-auto px-6 py-5">
          {error && (
            <div className="mb-5 flex items-start gap-3 rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-300"><AlertCircle size={17} className="mt-0.5 shrink-0" /> {error}</div>
          )}
          <div className="grid gap-5 sm:grid-cols-2">
            <Field label="Event" hint="The permanent context and memory boundary.">
              <select className={INPUT} value={form.event_id} disabled={Boolean(editingId)} onChange={(event) => set({ event_id: event.target.value })}>
                <option value="">Choose an Event</option>
                {dedicatedEvents.map((event) => <option key={event.id} value={event.id}>{event.title}</option>)}
              </select>
            </Field>
            <Field label="Schedule name" hint="A recognizable reason Aloy wakes.">
              <input className={INPUT} value={form.name} onChange={(event) => set({ name: event.target.value })} placeholder="Morning university check" />
            </Field>
          </div>

          <Field className="mt-5" label="What should Aloy do each time?" hint="Use the outcome you want; the Event already supplies its trusted context.">
            <textarea className={`${INPUT} min-h-28 resize-y`} value={form.task} onChange={(event) => set({ task: event.target.value })} placeholder="Review my timetable, upcoming exams, and course deadlines. Tell me what needs attention this week." />
          </Field>

          <div className="mt-6 border-t border-zinc-800 pt-5">
            <h3 className="font-display text-base font-semibold text-zinc-200">When should it run?</h3>
            <div className="mt-4 grid gap-4 sm:grid-cols-2">
              <Field label="Repeat">
                <select className={INPUT} value={form.cadence} onChange={(event) => set({ cadence: event.target.value as Cadence })}>
                  <option value="daily">Every day</option>
                  <option value="weekdays">Weekdays</option>
                  <option value="weekly">Once a week</option>
                  <option value="advanced">Advanced expression</option>
                </select>
              </Field>
              {form.cadence === 'advanced' ? (
                <Field label="Cron or interval" hint="Example: 0 7 * * 1-5">
                  <input className={INPUT} value={form.advanced} onChange={(event) => set({ advanced: event.target.value })} />
                </Field>
              ) : (
                <Field label="Time">
                  <input className={INPUT} type="time" value={form.time} onChange={(event) => set({ time: event.target.value })} />
                </Field>
              )}
              {form.cadence === 'weekly' && (
                <Field label="Day">
                  <select className={INPUT} value={form.weekday} onChange={(event) => set({ weekday: event.target.value })}>
                    {WEEKDAYS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
                  </select>
                </Field>
              )}
              <Field label="Timezone" hint="IANA timezone; daylight-saving changes are handled.">
                <input className={INPUT} value={form.timezone} onChange={(event) => set({ timezone: event.target.value })} placeholder="Africa/Johannesburg" />
              </Field>
            </div>
          </div>

          <div className="mt-6 border-t border-zinc-800 pt-5">
            <h3 className="font-display text-base font-semibold text-zinc-200">What may Aloy do unattended?</h3>
            <p className="mt-1 text-sm text-zinc-500">This authority is frozen into each queued run. Editing it later cannot broaden work already waiting.</p>
            <div className="mt-4 grid gap-3 sm:grid-cols-2">
              <AuthorityChoice selected={form.authority === 'report_only'} title="Read and report" description="Research, review connected information, and return findings without changing Tasks, files, Surface data, or drafts." onClick={() => set({ authority: 'report_only' })} />
              <AuthorityChoice selected={form.authority === 'organize'} title="Organize this Event" description="May update Event Tasks and prepare reversible drafts. Sending, booking, and other consequential actions still require approval." onClick={() => set({ authority: 'organize' })} />
            </div>
          </div>

          <div className="mt-6 grid gap-4 border-t border-zinc-800 pt-5 sm:grid-cols-2">
            <Field label="Notify me">
              <select className={INPUT} value={form.notification_mode} onChange={(event) => set({ notification_mode: event.target.value as ScheduleCreate['notification_mode'] })}>
                <option value="attention">Only when attention is needed</option>
                <option value="always">After every run</option>
              </select>
            </Field>
            <Field label="Maximum steps per run" hint="A hard ceiling, not a target.">
              <input className={INPUT} type="number" min={1} max={10000} value={form.max_steps} onChange={(event) => set({ max_steps: Number(event.target.value) || 15 })} />
            </Field>
          </div>

          <div className="mt-6 flex flex-col-reverse gap-3 border-t border-zinc-800 pt-5 sm:flex-row sm:items-center sm:justify-between">
            <p className="flex items-center gap-2 text-xs text-zinc-500"><ShieldCheck size={15} className="text-accent-700" /> MCP tools stay off for unattended runs until their authority is verifiable.</p>
            <Button onClick={() => void save()} disabled={!canSave || saving} className="shrink-0"><Save size={16} /> {saving ? 'Saving…' : editingId ? 'Save changes' : 'Create schedule'}</Button>
          </div>
        </div>
      </Modal>
    </div>
  );
}

function Field({ label, hint, className = '', children }: { label: string; hint?: string; className?: string; children: ReactNode }) {
  return (
    <label className={`block ${className}`}>
      <span className="mb-1.5 block text-sm font-medium text-zinc-300">{label}</span>
      {children}
      {hint && <span className="mt-1.5 block text-xs leading-5 text-zinc-600">{hint}</span>}
    </label>
  );
}

function AuthorityChoice({ selected, title, description, onClick }: { selected: boolean; title: string; description: string; onClick: () => void }) {
  return (
    <button type="button" aria-pressed={selected} onClick={onClick} className={`rounded-xl border p-4 text-left transition ${selected ? 'border-accent-600 bg-accent-950/35' : 'border-zinc-800 bg-zinc-900/40 hover:border-zinc-700'}`}>
      <span className="flex items-center gap-2 text-sm font-semibold text-zinc-200"><span className={`h-2.5 w-2.5 rounded-full ${selected ? 'bg-accent-600' : 'bg-zinc-700'}`} />{title}</span>
      <span className="mt-2 block text-xs leading-5 text-zinc-500">{description}</span>
    </button>
  );
}

function TrustLine({ children }: { children: ReactNode }) {
  return <p className="flex items-center gap-3 text-sm text-zinc-400"><ShieldCheck size={16} className="shrink-0 text-accent-700" /> {children}</p>;
}

function Fact({ label, value, detail, icon }: { label: string; value: string; detail: string; icon: ReactNode }) {
  return (
    <div className="flex items-start gap-3">
      <span className="mt-0.5 text-zinc-600">{icon}</span>
      <div className="min-w-0"><p className="text-xs font-medium uppercase tracking-wide text-zinc-600">{label}</p><p className="mt-1 text-sm font-medium text-zinc-300">{value}</p><p className="mt-1 text-xs leading-5 text-zinc-600">{detail}</p></div>
    </div>
  );
}

function IconButton({ label, onClick, disabled, danger = false, children }: { label: string; onClick: () => void; disabled?: boolean; danger?: boolean; children: ReactNode }) {
  return <button type="button" aria-label={label} title={label} disabled={disabled} onClick={onClick} className={`rounded-lg p-2 transition disabled:opacity-40 ${danger ? 'text-zinc-500 hover:bg-red-500/10 hover:text-red-400' : 'text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200'}`}>{children}</button>;
}

function RunHistory({ history, timezone }: { history: RunHistoryState | undefined; timezone: string }) {
  if (!history || history.loading) return <div className="flex items-center gap-3 border-t border-zinc-800 px-6 py-5 text-sm text-zinc-500"><Spinner className="h-4 w-4" /> Loading run receipts…</div>;
  if (history.error) return <div className="border-t border-zinc-800 px-6 py-5 text-sm text-red-400">{history.error}</div>;
  if (history.runs.length === 0) return <div className="border-t border-zinc-800 px-6 py-5 text-sm text-zinc-500">This schedule has not run yet.</div>;
  return (
    <div className="divide-y divide-zinc-800 border-t border-zinc-800">
      {history.runs.map((run) => (
        <div key={run.id} className="grid gap-2 px-5 py-4 sm:grid-cols-[160px_150px_minmax(0,1fr)] sm:px-6">
          <div><p className={`text-sm font-medium ${run.status === 'failed' || (run.status === 'completed' && !run.success) ? 'text-amber-400' : 'text-zinc-300'}`}>{runStatus(run)}</p><p className="mt-1 text-xs text-zinc-600">{run.steps_taken} steps</p></div>
          <p className="text-xs leading-5 text-zinc-500">{formatWhen(run.created_at, timezone)}</p>
          <p className="line-clamp-3 text-sm leading-5 text-zinc-500">{run.final_answer || (run.status === 'running' ? 'Aloy is working inside the Event.' : 'No written outcome was recorded.')}</p>
        </div>
      ))}
    </div>
  );
}
