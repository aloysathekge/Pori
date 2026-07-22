import { useCallback, useEffect, useState } from 'react';
import { ArchiveRestore, ArrowLeft, Trash2 } from 'lucide-react';
import { Link } from 'react-router-dom';
import {
  listEvents,
  permanentlyDeleteEvent,
  updateEvent,
  type EventSummary,
} from '@/api/events';
import { EventCover } from '@/components/events/EventCover';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Modal } from '@/components/ui/Modal';
import { Spinner } from '@/components/ui/Spinner';
import { useToast } from '@/contexts/toast';

export function ArchivedEventsPage() {
  const { showToast } = useToast();
  const [events, setEvents] = useState<EventSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [busyId, setBusyId] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<EventSummary | null>(null);
  const [confirmation, setConfirmation] = useState('');

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      setEvents((await listEvents('archived')).filter((event) => !event.is_life));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Archived Events could not be loaded.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    // The route load owns the initial remote state for this page.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void load();
  }, [load]);

  async function restore(event: EventSummary) {
    setBusyId(event.id);
    setError('');
    try {
      await updateEvent(event.id, { lifecycle: 'active' });
      setEvents((current) => current.filter((item) => item.id !== event.id));
      showToast({
        tone: 'success',
        title: `${event.title} restored`,
        description: 'It is active and visible in your sidebar again.',
      });
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'The Event could not be restored.');
    } finally {
      setBusyId(null);
    }
  }

  async function erase() {
    if (!deleteTarget || confirmation !== deleteTarget.title) return;
    setBusyId(deleteTarget.id);
    setError('');
    try {
      const result = await permanentlyDeleteEvent(deleteTarget.id, confirmation);
      setEvents((current) => current.filter((item) => item.id !== deleteTarget.id));
      showToast({
        tone: 'success',
        title: `${deleteTarget.title} permanently deleted`,
        description: result.storage_cleanup === 'complete'
          ? 'Its conversations, tasks, files, memory, Trail, and Surface data were erased.'
          : 'Its records were erased. Unreachable stored objects are awaiting infrastructure cleanup.',
      });
      setDeleteTarget(null);
      setConfirmation('');
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'The Event could not be permanently deleted.');
    } finally {
      setBusyId(null);
    }
  }

  return (
    <div className="h-full overflow-y-auto bg-zinc-950">
      <div className="mx-auto w-full max-w-5xl px-4 py-6 sm:px-6 sm:py-10">
        <Link to="/today" className="inline-flex min-h-11 items-center gap-2 text-sm text-zinc-500 hover:text-zinc-200">
          <ArrowLeft size={16} /> Back to Today
        </Link>
        <div className="mt-4">
          <p className="text-xs font-semibold uppercase tracking-[0.12em] text-zinc-500">Event library</p>
          <h1 className="mt-1 font-display text-3xl font-semibold tracking-tight text-zinc-100">Archived Events</h1>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-zinc-500">
            Archived Events are paused and hidden from your normal workspace. Restore one anytime, or permanently delete it after review.
          </p>
        </div>

        {error && <div role="alert" className="mt-5 rounded-xl border border-red-500/25 bg-red-500/10 px-4 py-3 text-sm text-red-400">{error}</div>}

        {loading ? (
          <div className="flex min-h-64 items-center justify-center"><Spinner className="h-7 w-7" /></div>
        ) : events.length === 0 ? (
          <section className="mt-8 rounded-2xl border border-dashed border-zinc-800 bg-zinc-900/35 px-5 py-14 text-center">
            <ArchiveRestore className="mx-auto text-zinc-600" size={28} />
            <h2 className="mt-3 text-base font-semibold text-zinc-200">Nothing archived</h2>
            <p className="mt-1 text-sm text-zinc-500">Events you archive will wait safely here.</p>
          </section>
        ) : (
          <div className="mt-8 grid gap-3">
            {events.map((event) => (
              <article key={event.id} className="flex flex-col gap-4 rounded-2xl border border-zinc-800 bg-zinc-900/55 p-4 sm:flex-row sm:items-center">
                <EventCover event={event} className="h-20 w-full shrink-0 rounded-xl border border-zinc-800 sm:w-28" />
                <div className="min-w-0 flex-1">
                  <h2 className="truncate text-base font-semibold text-zinc-100">{event.title}</h2>
                  <p className="mt-1 line-clamp-2 text-sm leading-5 text-zinc-500">{event.summary || 'No Event summary.'}</p>
                  <p className="mt-2 text-xs text-zinc-600">Archived workspace · Created {new Date(event.created_at).toLocaleDateString()}</p>
                </div>
                <div className="flex shrink-0 gap-2">
                  <Button variant="secondary" size="sm" disabled={busyId === event.id} onClick={() => void restore(event)}>
                    <ArchiveRestore size={15} /> Restore
                  </Button>
                  <Button variant="ghost" size="sm" disabled={busyId === event.id} className="text-red-400 hover:bg-red-500/10 hover:text-red-300" onClick={() => { setDeleteTarget(event); setConfirmation(''); setError(''); }}>
                    <Trash2 size={15} /> Delete
                  </Button>
                </div>
              </article>
            ))}
          </div>
        )}
      </div>

      <Modal open={deleteTarget !== null} onClose={() => { if (!busyId) { setDeleteTarget(null); setConfirmation(''); } }} title="Permanently delete Event?">
        {deleteTarget && (
          <div>
            <div className="rounded-xl border border-red-500/25 bg-red-500/10 p-3.5 text-sm leading-6 text-red-200">
              This cannot be undone. Aloy will erase this Event’s conversation, tasks, files, memory, Trail, schedules, receipts, and Surface data.
            </div>
            {error && <div role="alert" className="mt-3 rounded-xl border border-red-500/25 bg-red-500/10 px-3 py-2 text-sm text-red-300">{error}</div>}
            <p className="mt-4 text-sm text-zinc-400">Type <strong className="text-zinc-200">{deleteTarget.title}</strong> to confirm.</p>
            <Input label="Event name" value={confirmation} onChange={(event) => setConfirmation(event.target.value)} autoComplete="off" className="mt-2" />
            <div className="mt-5 flex justify-end gap-2">
              <Button variant="ghost" onClick={() => { setDeleteTarget(null); setConfirmation(''); }} disabled={busyId !== null}>Cancel</Button>
              <Button variant="danger" onClick={() => void erase()} disabled={confirmation !== deleteTarget.title || busyId !== null}>
                {busyId ? <Spinner className="h-4 w-4" /> : <Trash2 size={16} />}
                Permanently delete
              </Button>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
}
