import { useCallback, useEffect, useState } from 'react';
import {
  ArrowLeft,
  CheckCircle2,
  Circle,
  FileText,
  MessageSquarePlus,
  Plus,
  Trash2,
} from 'lucide-react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { createConversation } from '@/api/conversations';
import {
  createEventTask,
  decideEventProposal,
  deleteEventTask,
  getEventSurface,
  updateEventTask,
  type EventSurfaceResponse,
} from '@/api/events';
import { ProposalCard } from '@/components/events/ProposalCard';
import { Button } from '@/components/ui/Button';
import { Spinner } from '@/components/ui/Spinner';

const INPUT =
  'w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 placeholder-zinc-500 focus:border-accent-500 focus:outline-none';

function formatDate(value: string) {
  return new Date(value).toLocaleString();
}

export function EventPage() {
  const { eventId = '' } = useParams();
  const navigate = useNavigate();
  const [data, setData] = useState<EventSurfaceResponse | null>(null);
  const [taskTitle, setTaskTitle] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const load = useCallback(async () => {
    if (!eventId) return;
    try {
      setData(await getEventSurface(eventId));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [eventId]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    load();
  }, [load]);

  async function addTask() {
    if (!taskTitle.trim()) return;
    setBusy(true);
    setError('');
    try {
      await createEventTask(eventId, taskTitle.trim());
      setTaskTitle('');
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function toggleTask(taskId: string, status: 'open' | 'done') {
    setError('');
    try {
      await updateEventTask(eventId, taskId, { status: status === 'open' ? 'done' : 'open' });
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function removeTask(taskId: string) {
    setError('');
    try {
      await deleteEventTask(eventId, taskId);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function decide(proposalId: string, decision: 'approve' | 'reject') {
    setError('');
    try {
      await decideEventProposal(eventId, proposalId, decision);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function newSession() {
    if (!data) return;
    setBusy(true);
    setError('');
    try {
      const conversation = await createConversation({
        title: data.event.title,
        event_id: data.event.id,
      });
      navigate(`/chat/${conversation.id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  if (!data) {
    return (
      <div className="flex h-full items-center justify-center">
        {error ? <p className="text-sm text-red-600">{error}</p> : <Spinner />}
      </div>
    );
  }

  const status = data.surface.sections.find((section) => section.kind === 'status');
  const tasks = data.surface.sections.find((section) => section.kind === 'tasks');
  const activity = data.surface.sections.find((section) => section.kind === 'activity');
  const notes = data.surface.sections.find((section) => section.kind === 'notes');
  const files = data.surface.sections.find((section) => section.kind === 'files');

  return (
    <div className="mx-auto min-h-full max-w-6xl px-5 py-8 lg:px-10">
      <Link to="/today" className="inline-flex items-center gap-1 text-sm text-zinc-500 hover:text-zinc-300">
        <ArrowLeft size={15} /> Today
      </Link>

      <header className="mt-5 flex flex-col justify-between gap-5 sm:flex-row sm:items-start">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <h1 className="font-display text-3xl font-semibold tracking-tight text-zinc-100">
              {data.event.title}
            </h1>
            <span className="rounded-full border border-zinc-700 px-2.5 py-1 text-xs capitalize text-zinc-400">
              {data.event.is_life ? 'Life' : data.event.phase || 'Project'}
            </span>
          </div>
          {status?.kind === 'status' && status.summary && (
            <p className="mt-2 max-w-2xl text-sm text-zinc-400">{status.summary}</p>
          )}
        </div>
        <Button onClick={newSession} disabled={busy}>
          <MessageSquarePlus size={16} /> New session here
        </Button>
      </header>

      {error && (
        <div className="mt-5 rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-2 text-sm text-red-600">
          {error}
        </div>
      )}

      {data.surface.proposals.length > 0 && (
        <section className="mt-8">
          <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-zinc-500">
            Needs your decision
          </h2>
          <div className="grid gap-3 lg:grid-cols-2">
            {data.surface.proposals.map((proposal) => (
              <ProposalCard
                key={proposal.id}
                proposal={proposal}
                onDecision={(decision) => decide(proposal.id, decision)}
              />
            ))}
          </div>
        </section>
      )}

      <div className="mt-8 grid gap-6 lg:grid-cols-[1.15fr_0.85fr]">
        <div className="space-y-6">
          <section className="rounded-2xl border border-zinc-800 bg-zinc-900 p-5">
            <h2 className="font-display text-lg font-semibold text-zinc-100">Tasks</h2>
            <div className="mt-4 flex gap-2">
              <input
                className={INPUT}
                value={taskTitle}
                onChange={(event) => setTaskTitle(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') addTask();
                }}
                placeholder="Add a durable task"
              />
              <Button size="icon" onClick={addTask} disabled={busy || !taskTitle.trim()} aria-label="Add task">
                <Plus size={17} />
              </Button>
            </div>
            <div className="mt-4 divide-y divide-zinc-800">
              {tasks?.kind === 'tasks' && tasks.tasks.map((task) => (
                <div key={task.id} className="group flex items-center gap-3 py-3">
                  <button
                    type="button"
                    onClick={() => toggleTask(task.id, task.status)}
                    className="text-zinc-500 hover:text-accent-600"
                    aria-label={task.status === 'open' ? 'Complete task' : 'Reopen task'}
                  >
                    {task.status === 'done' ? <CheckCircle2 size={18} /> : <Circle size={18} />}
                  </button>
                  <span className={`flex-1 text-sm ${task.status === 'done' ? 'text-zinc-600 line-through' : 'text-zinc-300'}`}>
                    {task.title}
                  </span>
                  <button
                    type="button"
                    onClick={() => removeTask(task.id)}
                    className="rounded p-1 text-zinc-700 opacity-0 hover:text-red-500 group-hover:opacity-100 focus:opacity-100"
                    aria-label="Delete task"
                  >
                    <Trash2 size={15} />
                  </button>
                </div>
              ))}
              {tasks?.kind === 'tasks' && tasks.tasks.length === 0 && (
                <p className="py-5 text-sm text-zinc-500">No tasks yet.</p>
              )}
            </div>
          </section>

          <section className="rounded-2xl border border-zinc-800 bg-zinc-900 p-5">
            <h2 className="font-display text-lg font-semibold text-zinc-100">Trail</h2>
            <p className="mt-1 text-xs text-zinc-500">The complete durable history for this Event.</p>
            <div className="mt-4 space-y-4">
              {activity?.kind === 'activity' && activity.entries.map((entry) => (
                <div key={entry.id} className="relative border-l border-zinc-700 pl-4">
                  <span className="absolute -left-1 top-1 h-2 w-2 rounded-full bg-zinc-600" />
                  <p className="text-sm text-zinc-300">{entry.summary}</p>
                  <p className="mt-1 text-xs text-zinc-600">{formatDate(entry.created_at)}</p>
                  {entry.evidence_refs.length > 0 && (
                    <p className="mt-1 text-xs font-medium text-accent-700">Receipt-backed evidence attached</p>
                  )}
                </div>
              ))}
            </div>
          </section>
        </div>

        <aside className="space-y-6">
          <section className="rounded-2xl border border-zinc-800 bg-zinc-900 p-5">
            <h2 className="font-display text-lg font-semibold text-zinc-100">Notes</h2>
            <p className="mt-3 whitespace-pre-wrap text-sm leading-6 text-zinc-400">
              {notes?.kind === 'notes' && notes.notes ? notes.notes : 'No project notes yet.'}
            </p>
          </section>

          <section className="rounded-2xl border border-zinc-800 bg-zinc-900 p-5">
            <h2 className="font-display text-lg font-semibold text-zinc-100">Files</h2>
            <p className="mt-1 text-xs text-zinc-500">Durable uploads and agent-created artifacts only.</p>
            <div className="mt-4 space-y-2">
              {files?.kind === 'files' && files.files.map((file) => (
                <div key={file.id} className="flex items-center gap-3 rounded-lg border border-zinc-800 bg-zinc-950/40 p-3">
                  <FileText size={17} className="shrink-0 text-zinc-500" />
                  <div className="min-w-0">
                    <p className="truncate text-sm text-zinc-300">{file.name}</p>
                    <p className="text-xs text-zinc-600">{file.kind} · {Math.max(1, Math.round(file.size_bytes / 1024))} KB</p>
                  </div>
                </div>
              ))}
              {files?.kind === 'files' && files.files.length === 0 && (
                <p className="text-sm text-zinc-500">No durable files yet.</p>
              )}
            </div>
          </section>
        </aside>
      </div>
    </div>
  );
}
