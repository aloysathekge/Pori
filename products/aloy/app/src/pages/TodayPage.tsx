import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ArrowRight,
  Bell,
  CalendarDays,
  CheckCircle2,
  Clock3,
  LoaderCircle,
  Play,
} from 'lucide-react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import {
  createEvent,
  decideEventProposal,
  getToday,
  workOnEventTask,
  type EventProposal,
  type EventTask,
  type TodayEventGroup,
  type TodayNotification,
  type TodayResponse,
} from '@/api/events';
import {
  createConversation as createLifeConversation,
  listConversations,
} from '@/api/conversations';
import { getProfile, updateProfile } from '@/api/profile';
import { useAuth } from '@/contexts/useAuth';
import { Button } from '@/components/ui/Button';
import { ChatIcon, EventIcon } from '@/components/icons';
import { Spinner } from '@/components/ui/Spinner';
import { formatRelativeTime } from '@/lib/time';
import type { ConversationResponse, UserProfileResponse } from '@/types';

const INPUT =
  'w-full rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-100 placeholder-zinc-500 focus:border-accent-500 focus:outline-none';

type TaskItem = {
  group: TodayEventGroup;
  task: EventTask;
};

const priorityWeight: Record<EventTask['priority'], number> = {
  urgent: 4,
  high: 3,
  normal: 2,
  low: 1,
};

function taskSort(left: TaskItem, right: TaskItem) {
  const priority = priorityWeight[right.task.priority] - priorityWeight[left.task.priority];
  if (priority !== 0) return priority;
  if (left.task.due_at && right.task.due_at) {
    return new Date(left.task.due_at).getTime() - new Date(right.task.due_at).getTime();
  }
  if (left.task.due_at) return -1;
  if (right.task.due_at) return 1;
  return new Date(left.task.created_at).getTime() - new Date(right.task.created_at).getTime();
}

function firstName(profile: UserProfileResponse | null, session: ReturnType<typeof useAuth>['session']) {
  const metadataName = session?.user.user_metadata.full_name || session?.user.user_metadata.name;
  const raw = profile?.display_name || metadataName;
  if (typeof raw === 'string' && raw.trim()) return raw.trim().split(/\s+/)[0]!;
  const emailName = session?.user.email?.split('@')[0]?.split(/[._-]/)[0];
  if (emailName) return emailName.charAt(0).toUpperCase() + emailName.slice(1);
  return 'there';
}

function greeting(now: Date) {
  const hour = now.getHours();
  if (hour < 12) return 'Good morning';
  if (hour < 18) return 'Good afternoon';
  return 'Good evening';
}

function dueLabel(task: EventTask) {
  if (!task.due_at) return null;
  return new Intl.DateTimeFormat(undefined, {
    weekday: 'short',
    day: 'numeric',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(task.due_at));
}

function uniqueTasks(items: TaskItem[]) {
  return Array.from(new Map(items.map((item) => [item.task.id, item])).values());
}

export function TodayPage() {
  const navigate = useNavigate();
  const { session } = useAuth();
  const [searchParams, setSearchParams] = useSearchParams();
  const [today, setToday] = useState<TodayResponse | null>(null);
  const [profile, setProfile] = useState<UserProfileResponse | null>(null);
  const [lifeConversations, setLifeConversations] = useState<ConversationResponse[]>([]);
  const [creating, setCreating] = useState(false);
  const [title, setTitle] = useState('');
  const [summary, setSummary] = useState('');
  const [saving, setSaving] = useState(false);
  const [creatingConversation, setCreatingConversation] = useState(false);
  const [taskActionId, setTaskActionId] = useState<string | null>(null);
  const [reviewingProposalId, setReviewingProposalId] = useState<string | null>(null);
  const [notificationsOpen, setNotificationsOpen] = useState(true);
  const [markingRead, setMarkingRead] = useState(false);
  const [error, setError] = useState('');

  const load = useCallback(async () => {
    setError('');
    try {
      const response = await getToday();
      setToday(response);
      const life = response.events.find((group) => group.event.is_life);
      const [loadedProfile, conversations] = await Promise.all([
        getProfile().catch(() => null),
        life ? listConversations(4, 0, life.event.id).catch(() => []) : [],
      ]);
      setProfile(loadedProfile);
      setLifeConversations(conversations);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- load external Today state on mount
    void load();
  }, [load]);

  const now = useMemo(() => new Date(), []);
  const userName = firstName(profile, session);
  const dateLabel = new Intl.DateTimeFormat(undefined, {
    weekday: 'long',
    day: 'numeric',
    month: 'long',
  }).format(now);
  const dedicatedGroups = useMemo(
    () => today?.events.filter((group) => !group.event.is_life) ?? [],
    [today],
  );

  const { needsTasks, workingTasks, comingUpTasks, quietGroups, pendingProposals } = useMemo(() => {
    const needs: TaskItem[] = [];
    const working: TaskItem[] = [];
    const upcoming: TaskItem[] = [];
    const proposals: Array<{ group: TodayEventGroup; proposal: EventProposal }> = [];
    const quiet: TodayEventGroup[] = [];

    for (const group of dedicatedGroups) {
      proposals.push(...group.needs_decision.map((proposal) => ({ group, proposal })));
      const blockedIds = new Set([...group.blocked, ...group.stale].map((task) => task.id));
      needs.push(...group.blocked.map((task) => ({ group, task })));
      needs.push(...group.stale.map((task) => ({ group, task })));

      for (const task of group.upcoming) {
        if (task.status === 'queued' || task.status === 'in_progress') {
          working.push({ group, task });
        } else if (
          !blockedIds.has(task.id) &&
          (task.priority === 'urgent' ||
            (task.due_at !== null && new Date(task.due_at).getTime() <= now.getTime()))
        ) {
          needs.push({ group, task });
        } else if (!blockedIds.has(task.id)) {
          upcoming.push({ group, task });
        }
      }

      if (group.needs_decision.length === 0 && group.upcoming.length === 0) quiet.push(group);
    }

    return {
      needsTasks: uniqueTasks(needs).sort(taskSort),
      workingTasks: uniqueTasks(working).sort(taskSort),
      comingUpTasks: uniqueTasks(upcoming).sort(taskSort),
      quietGroups: quiet,
      pendingProposals: proposals,
    };
  }, [dedicatedGroups, now]);

  const preferences = profile?.preferences ?? {};
  const notifications = today?.notifications ?? [];
  const notificationReadAt =
    typeof preferences.today_notifications_read_at === 'string'
      ? new Date(preferences.today_notifications_read_at).getTime()
      : 0;
  const unreadCount = notifications.filter(
    (notification) => new Date(notification.created_at).getTime() > notificationReadAt,
  ).length;
  const needsEventCount = new Set([
    ...needsTasks.map((item) => item.group.event.id),
    ...pendingProposals.map((item) => item.group.event.id),
  ]).size;
  const briefing =
    needsEventCount > 0 || workingTasks.length > 0
      ? `${needsEventCount || 'No'} Event${needsEventCount === 1 ? '' : 's'} need${needsEventCount === 1 ? 's' : ''} you; Aloy is working on ${workingTasks.length} item${workingTasks.length === 1 ? '' : 's'}.`
      : 'Nothing urgent. Your Events are ready when you are.';
  const showEventCreator = creating || searchParams.get('new') === 'event';

  function toggleEventCreator() {
    if (searchParams.get('new') === 'event') {
      setSearchParams({});
      setCreating(false);
      return;
    }
    setCreating((value) => !value);
  }

  async function startConversation() {
    setCreatingConversation(true);
    setError('');
    try {
      const conversation = await createLifeConversation({});
      navigate(`/chat/${conversation.id}`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not start a conversation');
    } finally {
      setCreatingConversation(false);
    }
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
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setSaving(false);
    }
  }

  async function workOnTask(item: TaskItem) {
    setTaskActionId(item.task.id);
    setError('');
    try {
      await workOnEventTask(item.group.event.id, item.task.id);
      await load();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setTaskActionId(null);
    }
  }

  async function decide(eventId: string, proposalId: string, decision: 'approve' | 'reject') {
    setError('');
    try {
      await decideEventProposal(eventId, proposalId, decision);
      setReviewingProposalId(null);
      await load();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }

  async function markAllRead() {
    const timestamp = new Date().toISOString();
    setMarkingRead(true);
    try {
      const updated = await updateProfile({
        preferences: { ...preferences, today_notifications_read_at: timestamp },
      });
      setProfile(updated);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setMarkingRead(false);
    }
  }

  function openNotification(notification: TodayNotification) {
    if (notification.kind === 'approval_required' && notification.proposal_id) {
      setReviewingProposalId((current) =>
        current === notification.proposal_id ? null : notification.proposal_id,
      );
      return;
    }
    navigate(`/events/${notification.event_id}`);
  }

  function proposalFor(notification: TodayNotification) {
    if (!notification.proposal_id) return null;
    return (
      today?.events
        .flatMap((group) => group.needs_decision)
        .find((proposal) => proposal.id === notification.proposal_id) ?? null
    );
  }

  return (
    <div className="h-full overflow-y-auto bg-zinc-950">
      <div className="mx-auto min-h-full max-w-[1500px] px-5 py-7 lg:px-10 lg:py-10">
        <header className="flex flex-col gap-6 border-b border-zinc-800 pb-7 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <p className="text-sm font-medium text-zinc-400">{dateLabel}</p>
            <h1 className="mt-2 font-display text-3xl font-semibold tracking-tight text-zinc-100 sm:text-4xl">
              {greeting(now)}, {userName}
            </h1>
            <p className="mt-2 text-sm text-zinc-400">{briefing}</p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => setNotificationsOpen((value) => !value)}
              aria-label={`${unreadCount} unread notifications`}
              aria-expanded={notificationsOpen}
              className="relative flex h-10 w-10 items-center justify-center rounded-xl text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200"
            >
              <Bell size={19} />
              {unreadCount > 0 && (
                <span className="absolute -right-0.5 -top-0.5 flex h-5 min-w-5 items-center justify-center rounded-full bg-accent-600 px-1 text-[10px] font-semibold text-white">
                  {Math.min(unreadCount, 99)}
                </span>
              )}
            </button>
            <Button onClick={() => void startConversation()} disabled={creatingConversation}>
              {creatingConversation ? <LoaderCircle size={16} className="animate-spin" /> : <ChatIcon size={17} />}
              New conversation
            </Button>
            <Button variant="outline" onClick={toggleEventCreator} className="border-accent-700 text-accent-700">
              <EventIcon size={17} /> New Event
            </Button>
          </div>
        </header>

        {error && (
          <div className="mt-5 rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-600">
            {error}
          </div>
        )}

        {showEventCreator && (
          <section className="mt-5 rounded-2xl border border-zinc-800 bg-zinc-900 p-5">
            <div className="flex items-start gap-3">
              <span className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-accent-500/10 text-accent-700">
                <EventIcon size={18} />
              </span>
              <div>
                <h2 className="font-display text-lg font-semibold text-zinc-100">Start a durable Event</h2>
                <p className="mt-0.5 text-sm text-zinc-500">One continuous conversation with its own work, files, decisions, evidence, and Surface.</p>
              </div>
            </div>
            <div className="mt-4 grid gap-3 sm:grid-cols-[1fr_1.5fr_auto]">
              <input className={INPUT} value={title} onChange={(event) => setTitle(event.target.value)} placeholder="Event name" autoFocus />
              <input className={INPUT} value={summary} onChange={(event) => setSummary(event.target.value)} placeholder="What does success look like?" />
              <Button disabled={saving || !title.trim()} onClick={() => void saveEvent()}>
                {saving ? 'Creating…' : 'Create Event'}
              </Button>
            </div>
          </section>
        )}

        {!today ? (
          <div className="flex h-72 items-center justify-center"><Spinner /></div>
        ) : (
          <div className={`mt-6 grid items-start gap-7 ${notificationsOpen ? 'xl:grid-cols-[minmax(0,1fr)_360px]' : ''}`}>
            <main className="min-w-0 space-y-7">
              <section className="rounded-2xl border border-accent-700/20 bg-accent-50/45 p-5 sm:p-6">
                <div className="flex flex-col gap-5 sm:flex-row sm:items-center sm:justify-between">
                  <div className="flex min-w-0 items-center gap-3">
                    <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full border border-accent-700/20 bg-zinc-900 text-accent-700">
                      <ChatIcon size={19} />
                    </span>
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <h2 className="font-display text-xl font-semibold text-zinc-100">Life</h2>
                        <span className="text-sm text-zinc-500">Your personal space with Aloy</span>
                      </div>
                      {lifeConversations[0] ? (
                        <p className="mt-1 truncate text-sm text-zinc-400">
                          Recent conversation: {lifeConversations[0].title || 'New conversation'} · {formatRelativeTime(lifeConversations[0].updated_at)}
                        </p>
                      ) : (
                        <p className="mt-1 text-sm text-zinc-400">Capture a thought, ask for help, or start something loose.</p>
                      )}
                    </div>
                  </div>
                  <div className="flex shrink-0 items-center gap-2">
                    {lifeConversations[0] && (
                      <Button variant="outline" size="sm" onClick={() => navigate(`/chat/${lifeConversations[0]!.id}`)}>Continue</Button>
                    )}
                    <button type="button" onClick={() => void startConversation()} className="text-sm font-medium text-accent-700 hover:text-accent-600">
                      Ask Aloy or capture a thought <ArrowRight size={14} className="ml-1 inline" />
                    </button>
                  </div>
                </div>
              </section>

              <section>
                <div className="flex items-end justify-between border-b border-zinc-800 pb-3">
                  <div>
                    <h2 className="font-display text-xl font-semibold text-zinc-100">Needs you</h2>
                    <p className="mt-1 text-sm text-zinc-500">Decisions, blockers, stale work, and urgent priorities.</p>
                  </div>
                  {(pendingProposals.length + needsTasks.length) > 0 && (
                    <span className="text-xs font-semibold text-red-500">{pendingProposals.length + needsTasks.length} waiting</span>
                  )}
                </div>
                <div className="divide-y divide-zinc-800">
                  {pendingProposals.map(({ group, proposal }) => (
                    <div key={proposal.id} className="grid gap-3 py-4 md:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)_auto] md:items-center">
                      <div className="flex min-w-0 items-start gap-3">
                        <span className="mt-1.5 h-2.5 w-2.5 shrink-0 rounded-full bg-red-500" />
                        <div className="min-w-0">
                          <p className="font-medium text-zinc-100">Approval requested</p>
                          <p className="mt-1 truncate text-sm text-zinc-500">{group.event.title} · {group.event.summary}</p>
                        </div>
                      </div>
                      <p className="text-sm text-zinc-400">{proposal.reason || proposal.impact}</p>
                      <Button variant="outline" size="sm" onClick={() => setReviewingProposalId(proposal.id)}>Review</Button>
                    </div>
                  ))}
                  {needsTasks.map((item) => (
                    <div key={item.task.id} className="grid gap-3 py-4 md:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)_auto] md:items-center">
                      <div className="flex min-w-0 items-start gap-3">
                        <span className="mt-1.5 h-2.5 w-2.5 shrink-0 rounded-full bg-amber-500" />
                        <div className="min-w-0">
                          <p className="font-medium text-zinc-100">{item.task.title}</p>
                          <p className="mt-1 truncate text-sm text-zinc-500">{item.group.event.title} · {item.group.event.summary}</p>
                        </div>
                      </div>
                      <div className="text-sm text-zinc-400">
                        <span className="font-medium text-zinc-300">Why now:</span>{' '}
                        {item.task.blocker || (item.group.stale.some((task) => task.id === item.task.id) ? `Last updated ${formatRelativeTime(item.task.updated_at)}.` : dueLabel(item.task) || 'Marked as a priority.')}
                      </div>
                      <Button
                        size="sm"
                        onClick={() => item.task.status === 'open' ? void workOnTask(item) : navigate(`/events/${item.group.event.id}`)}
                        disabled={taskActionId === item.task.id}
                      >
                        {taskActionId === item.task.id ? <LoaderCircle size={14} className="animate-spin" /> : <Play size={14} />}
                        {item.task.status === 'open' ? 'Work on this' : 'Open Event'}
                      </Button>
                    </div>
                  ))}
                  {pendingProposals.length === 0 && needsTasks.length === 0 && (
                    <div className="flex items-center gap-3 py-5 text-sm text-zinc-500">
                      <CheckCircle2 size={17} className="text-accent-700" /> Nothing needs your attention right now.
                    </div>
                  )}
                </div>
              </section>

              {workingTasks.length > 0 && (
                <section>
                  <div className="border-b border-zinc-800 pb-3">
                    <h2 className="font-display text-xl font-semibold text-zinc-100">Aloy is working</h2>
                    <p className="mt-1 text-sm text-zinc-500">Durable work continues even when you leave this screen.</p>
                  </div>
                  <div className="divide-y divide-zinc-800">
                    {workingTasks.map((item) => (
                      <button key={item.task.id} type="button" onClick={() => navigate(`/events/${item.group.event.id}`)} className="grid w-full gap-2 py-4 text-left md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto] md:items-center">
                        <div className="flex min-w-0 items-center gap-3">
                          <LoaderCircle size={16} className="shrink-0 animate-spin text-accent-700" />
                          <div className="min-w-0"><p className="truncate font-medium text-zinc-100">{item.task.title}</p><p className="mt-1 text-sm text-zinc-500">{item.group.event.title}</p></div>
                        </div>
                        <p className="truncate text-sm text-zinc-400">{item.task.instructions || 'Working toward the task definition of done.'}</p>
                        <span className="text-sm font-medium text-accent-700">{item.task.status === 'queued' ? 'Queued' : 'In progress'}</span>
                      </button>
                    ))}
                  </div>
                </section>
              )}

              <section>
                <div className="border-b border-zinc-800 pb-3">
                  <h2 className="font-display text-xl font-semibold text-zinc-100">Coming up</h2>
                  <p className="mt-1 text-sm text-zinc-500">The next meaningful work across your Events.</p>
                </div>
                <div className="divide-y divide-zinc-800">
                  {comingUpTasks.slice(0, 6).map((item) => (
                    <button key={item.task.id} type="button" onClick={() => navigate(`/events/${item.group.event.id}`)} className="grid w-full gap-2 py-3.5 text-left sm:grid-cols-[minmax(0,1fr)_auto_auto] sm:items-center">
                      <div className="flex min-w-0 items-center gap-3"><CalendarDays size={16} className="shrink-0 text-zinc-500" /><div className="min-w-0"><p className="truncate text-sm font-medium text-zinc-200">{item.task.title}</p><p className="mt-0.5 text-xs text-zinc-500">{item.group.event.title}</p></div></div>
                      <span className="text-sm text-zinc-500">{dueLabel(item.task) || 'No due date'}</span><ArrowRight size={15} className="text-accent-700" />
                    </button>
                  ))}
                  {comingUpTasks.length === 0 && quietGroups.map((group) => (
                    <button key={group.event.id} type="button" onClick={() => navigate(`/events/${group.event.id}`)} className="flex w-full items-center justify-between gap-4 py-4 text-left">
                      <div><p className="font-medium text-zinc-200">{group.event.title}</p><p className="mt-1 text-sm text-zinc-500">{group.event.summary || 'No work needs attention.'}</p></div>
                      <span className="flex items-center gap-1 text-sm font-medium text-accent-700">Open <ArrowRight size={14} /></span>
                    </button>
                  ))}
                  {comingUpTasks.length === 0 && quietGroups.length === 0 && (
                    <div className="flex items-center gap-3 py-5 text-sm text-zinc-500"><Clock3 size={16} /> Nothing scheduled yet.</div>
                  )}
                </div>
              </section>
            </main>

            {notificationsOpen && (
              <aside className="sticky top-6 overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-900">
                <div className="flex items-center justify-between border-b border-zinc-800 px-5 py-4">
                  <div className="flex items-center gap-2"><h2 className="font-display text-lg font-semibold text-zinc-100">Notifications</h2>{unreadCount > 0 && <span className="rounded-full bg-accent-600 px-2 py-0.5 text-[11px] font-semibold text-white">{unreadCount}</span>}</div>
                  <button type="button" onClick={() => void markAllRead()} disabled={markingRead || unreadCount === 0} className="text-xs font-medium text-accent-700 disabled:text-zinc-500">{markingRead ? 'Saving…' : 'Mark all read'}</button>
                </div>
                <div className="divide-y divide-zinc-800">
                  {notifications.slice(0, 8).map((notification) => {
                    const unread = new Date(notification.created_at).getTime() > notificationReadAt;
                    const proposal = proposalFor(notification);
                    const reviewing = notification.proposal_id === reviewingProposalId;
                    return (
                      <div key={notification.id} className="px-5 py-4">
                        <div className="flex items-start gap-3">
                          <span className={`mt-2 h-2 w-2 shrink-0 rounded-full ${unread ? 'bg-accent-600' : 'bg-zinc-700'}`} />
                          <div className="min-w-0 flex-1">
                            <div className="flex items-start justify-between gap-3"><p className="text-sm font-semibold text-zinc-100">{notification.title}</p><span className="shrink-0 text-xs text-zinc-500">{formatRelativeTime(notification.created_at)}</span></div>
                            <p className="mt-1 text-sm leading-5 text-zinc-400">{notification.summary}</p>
                            <div className="mt-2 flex items-center justify-between gap-3"><span className="text-xs font-medium text-accent-700">{notification.event_title}</span><button type="button" onClick={() => openNotification(notification)} className="text-xs font-medium text-accent-700 hover:text-accent-600">{notification.kind === 'approval_required' ? 'Review' : 'View'}</button></div>
                            {reviewing && proposal && (
                              <div className="mt-3 rounded-xl bg-zinc-950 p-3">
                                <p className="text-xs leading-5 text-zinc-400">{proposal.impact || proposal.reason}</p>
                                <div className="mt-3 flex gap-2"><Button size="sm" onClick={() => void decide(notification.event_id, proposal.id, 'approve')}>Approve</Button><Button variant="outline" size="sm" onClick={() => void decide(notification.event_id, proposal.id, 'reject')}>Reject</Button></div>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                  {notifications.length === 0 && (
                    <div className="px-5 py-10 text-center"><CheckCircle2 size={22} className="mx-auto text-accent-700" /><p className="mt-3 text-sm font-medium text-zinc-300">You are all caught up</p><p className="mt-1 text-xs text-zinc-500">Results, decisions, and meaningful changes will appear here.</p></div>
                  )}
                </div>
              </aside>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
