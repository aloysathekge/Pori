import { useCallback, useEffect, useState } from 'react';
import {
  Check,
  Globe2,
  History,
  LockKeyhole,
  PencilLine,
  RefreshCw,
  ShieldCheck,
  Trash2,
} from 'lucide-react';
import {
  correctEventMemory,
  forgetEventMemory,
  getEventMemory,
  promoteEventMemory,
  type EventMemoryRecord,
  type EventMemoryResponse,
} from '@/api/eventMemory';
import { Button } from '@/components/ui/Button';
import { MemoryIcon } from '@/components/icons';
import { Modal } from '@/components/ui/Modal';
import { Spinner } from '@/components/ui/Spinner';

interface EventMemoryPanelProps {
  eventId: string;
  refreshKey?: string;
}

type ConfirmIntent =
  | { kind: 'forget'; record: EventMemoryRecord }
  | { kind: 'promote'; record: EventMemoryRecord }
  | null;

function sourceLabel(source: string) {
  const labels: Record<string, string> = {
    agent: 'Remembered by Aloy',
    user: 'Added by you',
    user_correction: 'Corrected by you',
    user_promotion: 'Promoted by you',
    context_ingestion: 'From Event context',
    event_setup: 'From Event setup',
  };
  return labels[source] ?? source.replaceAll('_', ' ');
}

function formattedDate(value: string | null) {
  if (!value) return null;
  return new Date(value).toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });
}

function MemoryMeta({ record }: { record: EventMemoryRecord }) {
  const date = formattedDate(record.updated_at || record.created_at);
  return (
    <div className="mt-2 flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px] text-zinc-600">
      <span className="capitalize">{record.kind}</span>
      <span aria-hidden="true">·</span>
      <span>{sourceLabel(record.source)}</span>
      {date && <><span aria-hidden="true">·</span><span>{date}</span></>}
      {record.sensitivity !== 'internal' && (
        <span className="rounded-full border border-zinc-700 px-1.5 py-0.5 capitalize text-zinc-500">
          {record.sensitivity}
        </span>
      )}
    </div>
  );
}

export function EventMemoryPanel({ eventId, refreshKey }: EventMemoryPanelProps) {
  const [data, setData] = useState<EventMemoryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [actionId, setActionId] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [correction, setCorrection] = useState('');
  const [correctionReason, setCorrectionReason] = useState('');
  const [confirmIntent, setConfirmIntent] = useState<ConfirmIntent>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      setData(await getEventMemory(eventId));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setLoading(false);
    }
  }, [eventId]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- Event-scoped API load
    void load();
  }, [load, refreshKey]);

  function beginCorrection(record: EventMemoryRecord) {
    setEditingId(record.id);
    setCorrection(record.content);
    setCorrectionReason('');
    setError('');
    setNotice('');
  }

  function cancelCorrection() {
    setEditingId(null);
    setCorrection('');
    setCorrectionReason('');
  }

  async function saveCorrection(record: EventMemoryRecord) {
    const content = correction.trim();
    if (!content || content === record.content.trim()) return;
    setActionId(record.id);
    setError('');
    setNotice('');
    try {
      await correctEventMemory(
        eventId,
        record.id,
        content,
        correctionReason.trim() || undefined,
      );
      cancelCorrection();
      await load();
      setNotice('Memory corrected. The earlier version remains in the Event history.');
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setActionId(null);
    }
  }

  async function confirmAction() {
    if (!confirmIntent) return;
    const { kind, record } = confirmIntent;
    setActionId(record.id);
    setError('');
    setNotice('');
    try {
      if (kind === 'forget') {
        await forgetEventMemory(eventId, record.id);
        setNotice('Memory forgotten for this Event. Its audit history is retained.');
      } else {
        const result = await promoteEventMemory(eventId, record.id);
        setNotice(
          result.created
            ? 'Memory is now available across Aloy.'
            : 'This memory was already available across Aloy.',
        );
      }
      setConfirmIntent(null);
      await load();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setActionId(null);
    }
  }

  if (loading && !data) {
    return <div className="flex justify-center py-12"><Spinner className="h-6 w-6" /></div>;
  }

  const eventRecords = data?.event_records ?? [];
  const globalRecords = data?.inherited_global_records ?? [];

  return (
    <div className="space-y-5">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-zinc-200">What Aloy remembers here</h3>
          <p className="mt-1 text-xs leading-5 text-zinc-500">
            Accepted memory for this Event—not its Tasks, files, or complete transcript.
          </p>
        </div>
        <button
          type="button"
          onClick={() => void load()}
          disabled={loading}
          className="rounded-lg p-2 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-50"
          aria-label="Refresh Event memory"
          title="Refresh memory"
        >
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
        </button>
      </div>

      {error && (
        <div role="alert" className="rounded-lg border border-red-500/25 bg-red-500/10 px-3 py-2 text-xs leading-5 text-red-500">
          {error}
        </div>
      )}
      {notice && (
        <div role="status" className="flex gap-2 rounded-lg border border-emerald-500/20 bg-emerald-500/5 px-3 py-2 text-xs leading-5 text-emerald-500">
          <Check size={14} className="mt-0.5 shrink-0" />
          {notice}
        </div>
      )}

      <section aria-labelledby="event-memory-heading">
        <div className="mb-2 flex items-center justify-between">
          <h4 id="event-memory-heading" className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
            <MemoryIcon size={14} /> Remembered for this Event
          </h4>
          <span className="text-[11px] text-zinc-600">{data?.event_count ?? 0}</span>
        </div>
        <div className="space-y-2">
          {eventRecords.map((record) => (
            <article key={record.id} className="rounded-xl border border-zinc-800 bg-zinc-950/55 p-3">
              {editingId === record.id ? (
                <div className="space-y-2">
                  <label className="block text-xs font-medium text-zinc-300" htmlFor={`memory-correction-${record.id}`}>
                    Correct what Aloy should remember
                  </label>
                  <textarea
                    id={`memory-correction-${record.id}`}
                    value={correction}
                    onChange={(event) => setCorrection(event.target.value)}
                    rows={5}
                    maxLength={50_000}
                    autoFocus
                    className="w-full resize-y rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm leading-5 text-zinc-200 outline-none focus:border-accent-600"
                  />
                  <input
                    value={correctionReason}
                    onChange={(event) => setCorrectionReason(event.target.value)}
                    maxLength={1000}
                    placeholder="Why did this change? (optional)"
                    className="w-full rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-xs text-zinc-300 outline-none focus:border-accent-600"
                  />
                  <p className="flex gap-1.5 text-[11px] leading-4 text-zinc-600">
                    <History size={12} className="mt-0.5 shrink-0" />
                    The old version stays in the audit history and will no longer be recalled.
                  </p>
                  <div className="flex justify-end gap-2">
                    <Button size="sm" variant="ghost" onClick={cancelCorrection}>Cancel</Button>
                    <Button
                      size="sm"
                      onClick={() => void saveCorrection(record)}
                      disabled={actionId === record.id || !correction.trim() || correction.trim() === record.content.trim()}
                    >
                      {actionId === record.id ? <Spinner className="h-3 w-3" /> : <Check size={13} />}
                      Save correction
                    </Button>
                  </div>
                </div>
              ) : (
                <>
                  <p className="whitespace-pre-wrap text-sm leading-5 text-zinc-300">{record.content}</p>
                  <MemoryMeta record={record} />
                  <div className="mt-3 flex flex-wrap items-center gap-1 border-t border-zinc-800 pt-2">
                    {record.can_correct && (
                      <button type="button" onClick={() => beginCorrection(record)} className="flex items-center gap-1.5 rounded-md px-2 py-1 text-[11px] text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200">
                        <PencilLine size={12} /> Correct
                      </button>
                    )}
                    {record.can_promote ? (
                      <button type="button" onClick={() => setConfirmIntent({ kind: 'promote', record })} className="flex items-center gap-1.5 rounded-md px-2 py-1 text-[11px] text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200">
                        <Globe2 size={12} /> Use across Aloy
                      </button>
                    ) : record.promoted_global_id ? (
                      <span className="flex items-center gap-1.5 px-2 py-1 text-[11px] text-emerald-600">
                        <ShieldCheck size={12} /> Available across Aloy
                      </span>
                    ) : null}
                    {record.can_forget && (
                      <button type="button" onClick={() => setConfirmIntent({ kind: 'forget', record })} className="ml-auto flex items-center gap-1.5 rounded-md px-2 py-1 text-[11px] text-zinc-600 hover:bg-red-500/10 hover:text-red-500">
                        <Trash2 size={12} /> Forget
                      </button>
                    )}
                  </div>
                </>
              )}
            </article>
          ))}
          {eventRecords.length === 0 && (
            <div className="rounded-xl border border-dashed border-zinc-800 px-4 py-8 text-center">
              <MemoryIcon size={20} className="mx-auto text-zinc-700" />
              <p className="mt-2 text-sm text-zinc-400">No accepted Event memories yet.</p>
              <p className="mt-1 text-xs leading-5 text-zinc-600">
                Aloy can still use this Event&apos;s current Tasks, files, Trail, and conversation.
              </p>
            </div>
          )}
        </div>
      </section>

      <section aria-labelledby="global-memory-heading">
        <div className="mb-2 flex items-center justify-between">
          <h4 id="global-memory-heading" className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
            <Globe2 size={14} /> Inherited across Aloy
          </h4>
          <span className="text-[11px] text-zinc-600">{data?.inherited_global_count ?? 0}</span>
        </div>
        <p className="mb-2 text-xs leading-5 text-zinc-600">
          Stable preferences and facts this Event may use. Manage them from global Memory.
        </p>
        <div className="space-y-2">
          {globalRecords.map((record) => (
            <article key={record.id} className="rounded-xl border border-zinc-800/80 bg-zinc-950/30 p-3">
              <div className="flex items-start gap-2.5">
                <LockKeyhole size={14} className="mt-0.5 shrink-0 text-zinc-600" />
                <div className="min-w-0">
                  <p className="whitespace-pre-wrap text-sm leading-5 text-zinc-400">{record.content}</p>
                  <MemoryMeta record={record} />
                </div>
              </div>
            </article>
          ))}
          {globalRecords.length === 0 && <p className="rounded-lg border border-dashed border-zinc-800 py-5 text-center text-xs text-zinc-600">No inherited global memory.</p>}
          {(data?.inherited_global_count ?? 0) > globalRecords.length && (
            <p className="text-center text-[11px] text-zinc-600">
              Showing {globalRecords.length} of {data?.inherited_global_count} global memories.
            </p>
          )}
        </div>
      </section>

      <Modal
        open={confirmIntent !== null}
        onClose={() => { if (!actionId) setConfirmIntent(null); }}
        title={confirmIntent?.kind === 'forget' ? 'Forget this memory?' : 'Use this across Aloy?'}
      >
        {confirmIntent && (
          <div>
            <div className="rounded-xl border border-zinc-800 bg-zinc-950/60 p-3 text-sm leading-6 text-zinc-300">
              {confirmIntent.record.content}
            </div>
            <p className="mt-3 text-sm leading-6 text-zinc-500">
              {confirmIntent.kind === 'forget'
                ? 'Aloy will stop recalling this in the Event. The audit record remains so the change is accountable.'
                : 'A separate global memory will be created. Aloy may then use it in Life and your other Events.'}
            </p>
            <div className="mt-5 flex justify-end gap-2">
              <Button variant="ghost" onClick={() => setConfirmIntent(null)} disabled={!!actionId}>Cancel</Button>
              <Button
                variant={confirmIntent.kind === 'forget' ? 'danger' : 'primary'}
                onClick={() => void confirmAction()}
                disabled={actionId === confirmIntent.record.id}
              >
                {actionId === confirmIntent.record.id
                  ? <Spinner className="h-4 w-4" />
                  : confirmIntent.kind === 'forget' ? <Trash2 size={15} /> : <Globe2 size={15} />}
                {confirmIntent.kind === 'forget' ? 'Forget memory' : 'Use across Aloy'}
              </Button>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
}
