import { useCallback, useEffect, useState } from 'react';
import { AlertTriangle, ArrowRight, CheckCircle2, Circle, Clock3, Sparkles } from 'lucide-react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import {
  createEvent,
  decideEventProposal,
  getToday,
  type TodayResponse,
} from '@/api/events';
import { ProposalCard } from '@/components/events/ProposalCard';
import { Button } from '@/components/ui/Button';
import { EventIcon } from '@/components/icons';
import { Spinner } from '@/components/ui/Spinner';

const INPUT =
  'w-full rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-100 placeholder-zinc-500 focus:border-accent-500 focus:outline-none';

function when(value: string) {
  return new Intl.RelativeTimeFormat(undefined, { numeric: 'auto' }).format(
    Math.round((new Date(value).getTime() - Date.now()) / 3_600_000),
    'hour',
  );
}

export function TodayPage() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const [today, setToday] = useState<TodayResponse | null>(null);
  const [creating, setCreating] = useState(false);
  const [title, setTitle] = useState('');
  const [summary, setSummary] = useState('');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');

  const load = useCallback(async () => {
    try {
      setToday(await getToday());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    load();
  }, [load]);

  const showEventCreator = creating || searchParams.get('new') === 'event';

  function toggleEventCreator() {
    if (searchParams.get('new') === 'event') {
      setSearchParams({});
      setCreating(false);
      return;
    }
    setCreating((value) => !value);
  }

  async function saveEvent() {
    if (!title.trim()) return;
    setSaving(true);
    setError('');
    try {
      const event = await createEvent({
        title: title.trim(),
        summary: summary.trim(),
        phase: 'planning',
      });
      navigate(`/events/${event.id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }

  async function decide(eventId: string, proposalId: string, decision: 'approve' | 'reject') {
    setError('');
    try {
      await decideEventProposal(eventId, proposalId, decision);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  const decisionCount =
    today?.events.reduce((total, group) => total + group.needs_decision.length, 0) ?? 0;
  const taskCount =
    today?.events.reduce((total, group) => total + group.upcoming.length, 0) ?? 0;
  const attentionCount =
    today?.events.reduce(
      (total, group) => total + group.blocked.length + group.stale.length,
      0,
    ) ?? 0;

  return (
    <div className="mx-auto min-h-full max-w-6xl px-5 py-8 lg:px-10">
      <header className="flex flex-col justify-between gap-5 sm:flex-row sm:items-end">
        <div>
          <p className="mb-2 flex items-center gap-2 text-sm font-medium text-accent-700">
            <Sparkles size={16} /> Your durable view
          </p>
          <h1 className="font-display text-3xl font-semibold tracking-tight text-zinc-100">
            Today
          </h1>
          <p className="mt-2 text-sm text-zinc-400">
            {decisionCount} decisions need you · {taskCount} open tasks · {attentionCount} need attention
          </p>
        </div>
        <Button
          onClick={toggleEventCreator}
          className="px-5"
        >
          <EventIcon size={16} /> New Event
        </Button>
      </header>

      {error && (
        <div className="mt-5 rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-2 text-sm text-red-600">
          {error}
        </div>
      )}

      {showEventCreator && (
        <section className="mt-6 rounded-2xl border border-zinc-800 bg-zinc-900 p-6">
          <p className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.14em] text-accent-300">
            <EventIcon size={14} /> Event workspace
          </p>
          <h2 className="font-display text-xl font-semibold text-zinc-100">Start a new Event</h2>
          <p className="mt-1 text-sm text-zinc-500">
            One continuous conversation, plus its tasks, files, decisions, and evidence.
          </p>
          <div className="mt-4 grid gap-3 sm:grid-cols-[1fr_1.5fr_auto]">
            <input
              className={INPUT}
              value={title}
              onChange={(event) => setTitle(event.target.value)}
              placeholder="Event name"
              autoFocus
            />
            <input
              className={INPUT}
              value={summary}
              onChange={(event) => setSummary(event.target.value)}
              placeholder="What does success look like?"
            />
            <Button disabled={saving || !title.trim()} onClick={saveEvent}>
              {saving ? 'Creating…' : 'Create event'}
            </Button>
          </div>
        </section>
      )}

      {!today ? (
        <div className="flex h-64 items-center justify-center"><Spinner /></div>
      ) : (
        <div className="mt-8 space-y-6">
          {today.events.map((group) => (
            <section key={group.event.id} className="overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-900">
              <div className="flex items-center justify-between border-b border-zinc-800 px-5 py-4">
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <h2 className="truncate font-display text-lg font-semibold text-zinc-100">
                      {group.event.title}
                    </h2>
                    {group.event.is_life && (
                      <span className="rounded-full bg-accent-600/15 px-2 py-0.5 text-[11px] font-medium text-accent-700">
                        Life
                      </span>
                    )}
                  </div>
                  {group.event.summary && (
                    <p className="mt-1 truncate text-sm text-zinc-500">{group.event.summary}</p>
                  )}
                </div>
                <Link
                  to={
                    group.event.is_life
                      ? group.event.conversation_id
                        ? `/chat/${group.event.conversation_id}`
                        : '/chat'
                      : `/events/${group.event.id}`
                  }
                  className="ml-4 flex shrink-0 items-center gap-1 text-sm font-medium text-accent-700 hover:text-accent-600"
                >
                  Open <ArrowRight size={15} />
                </Link>
              </div>

              <div className="grid gap-6 p-5 lg:grid-cols-2">
                <div>
                  <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-zinc-500">
                    Needs your decision
                  </h3>
                  <div className="space-y-3">
                    {group.needs_decision.map((proposal) => (
                      <ProposalCard
                        key={proposal.id}
                        proposal={proposal}
                        onDecision={(decision) => decide(group.event.id, proposal.id, decision)}
                      />
                    ))}
                    {group.needs_decision.length === 0 && (
                      <p className="flex items-center gap-2 text-sm text-zinc-500">
                        <CheckCircle2 size={16} /> Nothing waiting on you
                      </p>
                    )}
                  </div>
                </div>

                <div>
                  <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-zinc-500">
                    Open work
                  </h3>
                  {(group.blocked.length > 0 || group.stale.length > 0) && (
                    <div className="mb-4 space-y-2 rounded-xl border border-amber-500/20 bg-amber-500/5 p-3">
                      {group.blocked.map((task) => (
                        <div key={`blocked-${task.id}`} className="flex items-start gap-2 text-sm text-amber-500">
                          <AlertTriangle size={14} className="mt-0.5 shrink-0" />
                          <span><strong className="font-medium">Blocked:</strong> {task.title}</span>
                        </div>
                      ))}
                      {group.stale.map((task) => (
                        <div key={`stale-${task.id}`} className="flex items-start gap-2 text-sm text-zinc-400">
                          <Clock3 size={14} className="mt-0.5 shrink-0 text-amber-600" />
                          <span><strong className="font-medium text-zinc-300">Stale:</strong> {task.title}</span>
                        </div>
                      ))}
                    </div>
                  )}
                  <div className="space-y-2">
                    {group.upcoming.slice(0, 5).map((task) => (
                      <div key={task.id} className="flex items-start gap-2 text-sm text-zinc-300">
                        <Circle size={14} className="mt-0.5 shrink-0 text-zinc-600" />
                        {task.title}
                      </div>
                    ))}
                    {group.upcoming.length === 0 && (
                      <p className="text-sm text-zinc-500">No open tasks.</p>
                    )}
                  </div>
                  {group.activity.length > 0 && (
                    <div className="mt-5 border-t border-zinc-800 pt-4">
                      <p className="text-sm text-zinc-400">{group.activity[0]!.summary}</p>
                      <p className="mt-1 text-xs text-zinc-600">{when(group.activity[0]!.created_at)}</p>
                    </div>
                  )}
                </div>
              </div>
            </section>
          ))}
        </div>
      )}
    </div>
  );
}
