import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ArrowRight,
  Bell,
  CalendarDays,
  CheckCircle2,
  Clock3,
  ExternalLink,
  Inbox,
  LoaderCircle,
  Mail,
  Play,
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import {
  decideEventProposal,
  getToday,
  getTodayEmails,
  workOnEventTask,
  type EventProposal,
  type EventTask,
  type TodayEmailMessage,
  type TodayEmailsResponse,
  type TodayEventGroup,
  type TodayNotification,
  type TodayResponse,
  type TodayScheduledWork,
} from '@/api/events';
import { createConversation as createLifeConversation } from '@/api/conversations';
import { getProfile, updateProfile } from '@/api/profile';
import { useAuth } from '@/contexts/useAuth';
import { Button } from '@/components/ui/Button';
import { Modal } from '@/components/ui/Modal';
import { ChatIcon, EventIcon } from '@/components/icons';
import { Spinner } from '@/components/ui/Spinner';
import { formatRelativeTime } from '@/lib/time';
import type { UserProfileResponse } from '@/types';

type TaskItem = {
  group: TodayEventGroup;
  task: EventTask;
};

type ScheduledItem = {
  group: TodayEventGroup;
  schedule: TodayScheduledWork;
};

type WorkingItem =
  | { kind: 'task'; group: TodayEventGroup; task: EventTask }
  | { kind: 'schedule'; group: TodayEventGroup; schedule: TodayScheduledWork };

type AttentionItem =
  | { kind: 'proposal'; group: TodayEventGroup; proposal: EventProposal }
  | { kind: 'task'; group: TodayEventGroup; task: EventTask };

const priorityWeight: Record<EventTask['priority'], number> = {
  urgent: 4,
  high: 3,
  normal: 2,
  low: 1,
};

const unavailableEmails: TodayEmailsResponse = {
  status: 'unavailable',
  account_email: null,
  messages: [],
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

function senderName(value: string) {
  const name = value.split('<')[0]?.trim().replace(/^"|"$/g, '');
  return name || value;
}

function emailTime(message: TodayEmailMessage) {
  if (!message.received_at) return '';
  return formatRelativeTime(message.received_at);
}

function attentionKey(item: AttentionItem) {
  return item.kind === 'proposal' ? item.proposal.id : item.task.id;
}

function attentionTitle(item: AttentionItem) {
  if (item.kind === 'proposal') {
    return item.proposal.reason || item.proposal.impact || 'Review requested action';
  }
  return item.task.title;
}

function attentionReason(item: AttentionItem) {
  if (item.kind === 'proposal') {
    return item.proposal.impact || 'Aloy needs your approval before this action can continue.';
  }
  if (item.task.blocker) return item.task.blocker;
  return dueLabel(item.task) || 'This work is marked as a priority.';
}

export function TodayPage() {
  const navigate = useNavigate();
  const { session } = useAuth();
  const [today, setToday] = useState<TodayResponse | null>(null);
  const [emails, setEmails] = useState<TodayEmailsResponse | null>(null);
  const [profile, setProfile] = useState<UserProfileResponse | null>(null);
  const [creatingConversation, setCreatingConversation] = useState(false);
  const [taskActionId, setTaskActionId] = useState<string | null>(null);
  const [reviewingProposalId, setReviewingProposalId] = useState<string | null>(null);
  const [notificationsOpen, setNotificationsOpen] = useState(false);
  const [markingRead, setMarkingRead] = useState(false);
  const [refreshingEmails, setRefreshingEmails] = useState(false);
  const [error, setError] = useState('');

  const load = useCallback(async () => {
    setError('');
    try {
      const [response, loadedProfile] = await Promise.all([
        getToday(),
        getProfile().catch(() => null),
      ]);
      setToday(response);
      setProfile(loadedProfile);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }, []);

  const loadEmailBrief = useCallback(async (showRefreshing = false) => {
    if (showRefreshing) setRefreshingEmails(true);
    try {
      setEmails(await getTodayEmails());
    } catch {
      setEmails(unavailableEmails);
    } finally {
      if (showRefreshing) setRefreshingEmails(false);
    }
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- load external Today state on mount
    void load();
    void loadEmailBrief();
  }, [load, loadEmailBrief]);

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

  const { needsTasks, workingTasks, scheduledWork, comingUpTasks, quietGroups, pendingProposals } = useMemo(() => {
    const needs: TaskItem[] = [];
    const working: TaskItem[] = [];
    const upcoming: TaskItem[] = [];
    const proposals: Array<{ group: TodayEventGroup; proposal: EventProposal }> = [];
    const scheduled: ScheduledItem[] = [];
    const quiet: TodayEventGroup[] = [];

    for (const group of dedicatedGroups) {
      proposals.push(...group.needs_decision.map((proposal) => ({ group, proposal })));
      const eventSchedules = group.scheduled_work ?? [];
      scheduled.push(...eventSchedules.map((schedule) => ({ group, schedule })));
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

      if (
        group.needs_decision.length === 0 &&
        group.upcoming.length === 0 &&
        eventSchedules.length === 0
      ) quiet.push(group);
    }

    return {
      needsTasks: uniqueTasks(needs).sort(taskSort),
      workingTasks: uniqueTasks(working).sort(taskSort),
      scheduledWork: scheduled,
      comingUpTasks: uniqueTasks(upcoming).sort(taskSort),
      quietGroups: quiet,
      pendingProposals: proposals,
    };
  }, [dedicatedGroups, now]);

  const workingItems = useMemo<WorkingItem[]>(
    () => [
      ...workingTasks.map(({ group, task }) => ({ kind: 'task' as const, group, task })),
      ...scheduledWork.map(({ group, schedule }) => ({
        kind: 'schedule' as const,
        group,
        schedule,
      })),
    ],
    [scheduledWork, workingTasks],
  );

  const attentionItems = useMemo<AttentionItem[]>(
    () => [
      ...pendingProposals.map(({ group, proposal }) => ({
        kind: 'proposal' as const,
        group,
        proposal,
      })),
      ...needsTasks.map(({ group, task }) => ({ kind: 'task' as const, group, task })),
    ],
    [needsTasks, pendingProposals],
  );
  const primaryAttention = attentionItems[0] ?? null;
  const nextAttention = attentionItems.slice(1, 4);

  const preferences = profile?.preferences ?? {};
  const showTodaySuggestions = preferences.show_today_suggestions !== false;
  const notifications = today?.notifications ?? [];
  const notificationReadAt =
    typeof preferences.today_notifications_read_at === 'string'
      ? new Date(preferences.today_notifications_read_at).getTime()
      : 0;
  const unreadCount = notifications.filter(
    (notification) => new Date(notification.created_at).getTime() > notificationReadAt,
  ).length;
  const briefing = attentionItems.length > 0 || workingItems.length > 0
    ? `${attentionItems.length || 'No'} decision${attentionItems.length === 1 ? '' : 's'} or priorit${attentionItems.length === 1 ? 'y' : 'ies'} need${attentionItems.length === 1 ? 's' : ''} you. Aloy is working on ${workingItems.length} item${workingItems.length === 1 ? '' : 's'}.`
    : 'Nothing urgent. Your Events are ready when you are.';

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

  function actOnAttention(item: AttentionItem) {
    if (item.kind === 'proposal') {
      setReviewingProposalId(item.proposal.id);
      setNotificationsOpen(true);
      return;
    }
    if (item.task.status === 'open') {
      void workOnTask({ group: item.group, task: item.task });
      return;
    }
    navigate(`/events/${item.group.event.id}`);
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

  async function refreshEmails() {
    await loadEmailBrief(true);
  }

  function openNotification(notification: TodayNotification) {
    if (notification.kind === 'approval_required' && notification.proposal_id) {
      setReviewingProposalId((current) =>
        current === notification.proposal_id ? null : notification.proposal_id,
      );
      return;
    }
    setNotificationsOpen(false);
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
      <div className="mx-auto min-h-full max-w-[1320px] px-4 py-5 sm:px-5 sm:py-7 lg:px-10 lg:py-9">
        <header className="flex flex-col gap-4 border-b border-zinc-800 pb-5 sm:gap-6 sm:pb-7 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <p className="text-sm font-medium text-zinc-400">{dateLabel}</p>
            <h1 className="mt-2 font-display text-2xl font-semibold tracking-tight text-zinc-100 sm:text-4xl">
              {greeting(now)}, {userName}
            </h1>
            <p className="mt-2 text-sm text-zinc-400">{briefing}</p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => setNotificationsOpen(true)}
              aria-label={`${unreadCount} unread notifications`}
              className="relative flex h-11 w-11 items-center justify-center rounded-xl border border-transparent text-zinc-400 transition-colors hover:border-zinc-800 hover:bg-zinc-900 hover:text-zinc-200 sm:h-10 sm:w-10"
            >
              <Bell size={19} />
              {unreadCount > 0 && (
                <span className="absolute -right-0.5 -top-0.5 flex h-5 min-w-5 items-center justify-center rounded-full bg-accent-600 px-1 text-[10px] font-semibold text-white">
                  {Math.min(unreadCount, 99)}
                </span>
              )}
            </button>
            <div className="hidden items-center gap-2 sm:flex">
              <Button onClick={() => void startConversation()} disabled={creatingConversation}>
                {creatingConversation ? <LoaderCircle size={16} className="animate-spin" /> : <ChatIcon size={17} />}
                New conversation
              </Button>
              <Button variant="outline" onClick={() => navigate('/events/new')} className="border-accent-700/60 text-accent-700">
                <EventIcon size={17} /> New Event
              </Button>
            </div>
          </div>
        </header>

        {error && (
          <div className="mt-5 rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-600">
            {error}
          </div>
        )}

        {!today ? (
          <div className="flex h-72 items-center justify-center"><Spinner /></div>
        ) : (
          <main className="mt-7 space-y-8">
            <section>
              <h2 className="font-display text-lg font-semibold text-zinc-100">First, this needs you</h2>
              <div className="mt-3 overflow-hidden border-y border-zinc-800">
                {primaryAttention ? (
                  <div className="border-l-2 border-red-500 px-4 py-5 sm:px-6">
                    <div className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
                      <div className="min-w-0">
                        <p className="text-xs font-semibold uppercase tracking-wide text-red-500">
                          {primaryAttention.kind === 'proposal' ? 'Approval required' : 'Priority'} · {primaryAttention.group.event.title}
                        </p>
                        <h3 className="mt-2 font-display text-2xl font-semibold text-zinc-100">
                          {attentionTitle(primaryAttention)}
                        </h3>
                        <p className="mt-2 max-w-3xl text-sm leading-6 text-zinc-400">
                          <span className="font-medium text-zinc-300">Why now:</span> {attentionReason(primaryAttention)}
                        </p>
                      </div>
                      <Button
                        className="shrink-0"
                        onClick={() => actOnAttention(primaryAttention)}
                        disabled={primaryAttention.kind === 'task' && taskActionId === primaryAttention.task.id}
                      >
                        {primaryAttention.kind === 'task' && taskActionId === primaryAttention.task.id
                          ? <LoaderCircle size={15} className="animate-spin" />
                          : primaryAttention.kind === 'task' && <Play size={15} />}
                        {primaryAttention.kind === 'proposal' ? 'Review and decide' : primaryAttention.task.status === 'open' ? 'Work on this' : 'Open Event'}
                      </Button>
                    </div>
                  </div>
                ) : (
                  <div className="flex items-center gap-3 px-4 py-6 text-sm text-zinc-500">
                    <CheckCircle2 size={18} className="text-accent-700" /> Nothing needs your decision right now.
                  </div>
                )}
              </div>
            </section>

            {nextAttention.length > 0 && (
              <section>
                <h2 className="font-display text-lg font-semibold text-zinc-100">Then</h2>
                <div className="mt-2 divide-y divide-zinc-800 border-b border-zinc-800">
                  {nextAttention.map((item) => (
                    <div key={attentionKey(item)} className="grid gap-3 py-4 sm:grid-cols-[minmax(0,1fr)_minmax(180px,0.45fr)_auto] sm:items-center">
                      <div className="flex min-w-0 items-start gap-3">
                        <span className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-accent-50 text-accent-700">
                          <EventIcon size={16} />
                        </span>
                        <div className="min-w-0">
                          <p className="truncate font-medium text-zinc-100">{attentionTitle(item)}</p>
                          <p className="mt-1 text-sm text-zinc-500">{item.group.event.title}</p>
                        </div>
                      </div>
                      <p className="text-sm text-zinc-400">{attentionReason(item)}</p>
                      <button type="button" onClick={() => actOnAttention(item)} className="flex items-center gap-1 text-sm font-medium text-accent-700 hover:text-accent-600">
                        {item.kind === 'proposal' ? 'Review' : item.task.status === 'open' ? 'Work on this' : 'Open Event'} <ArrowRight size={14} />
                      </button>
                    </div>
                  ))}
                </div>
              </section>
            )}

            <section>
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="font-display text-lg font-semibold text-zinc-100">Aloy is working</h2>
                  <p className="mt-1 text-sm text-zinc-500">Durable work continues when you leave Today.</p>
                </div>
                {workingItems.length > 2 && (
                  <button type="button" onClick={() => navigate(`/events/${workingItems[2]!.group.event.id}`)} className="text-sm font-medium text-accent-700">View all working</button>
                )}
              </div>
              <div className="mt-3 grid border-y border-zinc-800 md:grid-cols-2 md:divide-x md:divide-zinc-800">
                {workingItems.slice(0, 2).map((item) => {
                  const title = item.kind === 'task' ? item.task.title : item.schedule.schedule_name;
                  const detail = item.kind === 'task'
                    ? item.task.instructions || 'Working toward the definition of done.'
                    : item.schedule.instruction || 'Running because its Event Schedule became due.';
                  const status = item.kind === 'task'
                    ? item.task.status === 'queued' ? 'Queued' : 'In progress'
                    : item.schedule.status === 'pending' ? 'Queued by schedule' : 'Scheduled run';
                  return (
                    <button key={item.kind === 'task' ? item.task.id : item.schedule.run_id} type="button" onClick={() => navigate(`/events/${item.group.event.id}`)} className="flex min-w-0 items-center gap-4 px-1 py-4 text-left md:px-5">
                      {item.kind === 'schedule'
                        ? <Clock3 size={17} className="shrink-0 text-accent-700" />
                        : <LoaderCircle size={17} className="shrink-0 animate-spin text-accent-700" />}
                      <div className="min-w-0 flex-1">
                        <p className="truncate font-medium text-zinc-100">{title}</p>
                        <p className="mt-1 truncate text-sm text-zinc-500">{item.group.event.title} · {detail}</p>
                      </div>
                      <span className="text-xs font-medium text-accent-700">{status}</span>
                    </button>
                  );
                })}
                {workingItems.length === 0 && (
                  <div className="flex items-center gap-3 px-1 py-5 text-sm text-zinc-500 md:col-span-2 md:px-5">
                    <CheckCircle2 size={17} className="text-accent-700" /> No background work is running.
                  </div>
                )}
              </div>
            </section>

            <section aria-busy={emails === null || refreshingEmails}>
              <div className="flex items-end justify-between gap-4">
                <div>
                  <h2 className="font-display text-lg font-semibold text-zinc-100">Important emails</h2>
                  <p className="mt-1 text-sm text-zinc-500">
                    {emails?.account_email ? `From ${emails.account_email}` : 'A bounded brief from your connected inbox.'}
                  </p>
                </div>
                {emails?.status === 'ready' && emails.messages.length > 0 && (
                  <a href="https://mail.google.com/mail/u/0/#inbox" target="_blank" rel="noreferrer" className="flex items-center gap-1 text-sm font-medium text-accent-700 hover:text-accent-600">
                    View inbox <ExternalLink size={13} />
                  </a>
                )}
              </div>

              <div className="mt-3 border-y border-zinc-800">
                {emails === null || refreshingEmails ? (
                  <div className="flex items-center gap-3 px-1 py-6 text-sm text-zinc-500">
                    <LoaderCircle size={17} className="animate-spin text-accent-700" /> Refreshing your email brief…
                  </div>
                ) : emails.status === 'not_connected' ? (
                  <div className="flex flex-col gap-4 px-1 py-6 sm:flex-row sm:items-center sm:justify-between">
                    <div className="flex items-start gap-3">
                      <Inbox size={20} className="mt-0.5 text-zinc-500" />
                      <div><p className="font-medium text-zinc-200">Connect Google to see important emails here</p><p className="mt-1 text-sm text-zinc-500">Aloy reads a bounded inbox brief; sending still requires your approval.</p></div>
                    </div>
                    <Button variant="outline" size="sm" onClick={() => navigate('/connections')}>Connect Google</Button>
                  </div>
                ) : emails.status === 'unavailable' ? (
                  <div className="flex flex-col gap-4 px-1 py-6 sm:flex-row sm:items-center sm:justify-between">
                    <div className="flex items-start gap-3"><Mail size={19} className="mt-0.5 text-zinc-500" /><div><p className="font-medium text-zinc-200">Your email brief could not refresh</p><p className="mt-1 text-sm text-zinc-500">Today is still available. Try the inbox again when you are ready.</p></div></div>
                    <Button variant="outline" size="sm" onClick={() => void refreshEmails()}>Try again</Button>
                  </div>
                ) : emails.messages.length === 0 ? (
                  <div className="flex items-center gap-3 px-1 py-6 text-sm text-zinc-500"><CheckCircle2 size={17} className="text-accent-700" /> No unread or important messages from the last seven days.</div>
                ) : (
                  <div className="divide-y divide-zinc-800">
                    {emails.messages.map((message) => (
                      <div key={message.id} className="grid gap-3 py-3.5 sm:grid-cols-[minmax(150px,0.45fr)_minmax(0,1fr)_auto_auto] sm:items-center">
                        <div className="flex min-w-0 items-center gap-3">
                          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-accent-50 text-accent-700"><Mail size={16} /></span>
                          <p className="truncate text-sm font-medium text-zinc-300">{senderName(message.sender)}</p>
                        </div>
                        <div className="min-w-0"><p className="truncate text-sm font-medium text-zinc-100">{message.subject}</p><p className="mt-1 truncate text-xs text-zinc-500">{message.snippet || 'Open the message to read more.'}</p></div>
                        <span className="text-xs text-zinc-500">{emailTime(message)}</span>
                        <a href={message.provider_url} target="_blank" rel="noreferrer" className="flex items-center gap-1 text-sm font-medium text-accent-700 hover:text-accent-600">Open <ExternalLink size={13} /></a>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </section>

            <section className="pb-6">
              <h2 className="font-display text-lg font-semibold text-zinc-100">Later today</h2>
              <div className="mt-2 divide-y divide-zinc-800 border-y border-zinc-800">
                {comingUpTasks.slice(0, 4).map((item) => (
                  <button key={item.task.id} type="button" onClick={() => navigate(`/events/${item.group.event.id}`)} className="grid w-full gap-2 py-3.5 text-left sm:grid-cols-[auto_minmax(0,1fr)_auto_auto] sm:items-center">
                    <CalendarDays size={16} className="text-zinc-500" />
                    <div className="min-w-0"><p className="truncate text-sm font-medium text-zinc-200">{item.task.title}</p><p className="mt-0.5 text-xs text-zinc-500">{item.group.event.title}</p></div>
                    <span className="text-sm text-zinc-500">{dueLabel(item.task) || 'No due date'}</span><ArrowRight size={15} className="text-accent-700" />
                  </button>
                ))}
                {comingUpTasks.length === 0 && showTodaySuggestions && quietGroups.slice(0, 3).map((group) => (
                  <button key={group.event.id} type="button" onClick={() => navigate(`/events/${group.event.id}`)} className="flex w-full items-center justify-between gap-4 py-3.5 text-left">
                    <div className="min-w-0"><p className="truncate text-sm font-medium text-zinc-200">{group.event.title}</p><p className="mt-1 truncate text-xs text-zinc-500">Quiet right now · {group.event.summary || 'No work needs attention.'}</p></div>
                    <ArrowRight size={15} className="text-accent-700" />
                  </button>
                ))}
                {comingUpTasks.length === 0 && (!showTodaySuggestions || quietGroups.length === 0) && (
                  <div className="flex items-center gap-3 py-5 text-sm text-zinc-500"><Clock3 size={16} /> Nothing scheduled yet.</div>
                )}
              </div>
            </section>
          </main>
        )}
      </div>

      <Modal
        open={notificationsOpen}
        onClose={() => { setNotificationsOpen(false); setReviewingProposalId(null); }}
        title={<span className="flex items-center gap-2">Notifications {unreadCount > 0 && <span className="rounded-full bg-accent-50 px-2 py-0.5 text-xs font-semibold text-accent-700">{unreadCount}</span>}</span>}
        headerActions={(
          <button type="button" onClick={() => void markAllRead()} disabled={markingRead || unreadCount === 0} className="px-2 py-1 text-xs font-medium text-accent-700 disabled:text-zinc-500">
            {markingRead ? 'Saving…' : 'Mark all read'}
          </button>
        )}
        panelClassName="max-w-xl"
      >
        <div className="-mx-6 -mb-6 max-h-[min(70vh,640px)] overflow-y-auto border-t border-zinc-800">
          {notifications.slice(0, 10).map((notification) => {
            const unread = new Date(notification.created_at).getTime() > notificationReadAt;
            const proposal = proposalFor(notification);
            const reviewing = notification.proposal_id === reviewingProposalId;
            return (
              <div key={notification.id} className="border-b border-zinc-800 px-4 py-4 last:border-b-0 sm:px-6">
                <div className="flex items-start gap-3">
                  <span className={`mt-2 h-2 w-2 shrink-0 rounded-full ${unread ? 'bg-accent-600' : 'bg-zinc-700'}`} />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-start justify-between gap-3"><p className="text-sm font-semibold text-zinc-100">{notification.title}</p><span className="shrink-0 text-xs text-zinc-500">{formatRelativeTime(notification.created_at)}</span></div>
                    <p className="mt-1 text-sm leading-5 text-zinc-400">{notification.summary}</p>
                    <div className="mt-2 flex items-center justify-between gap-3"><span className="text-xs font-medium text-accent-700">{notification.event_title}</span><button type="button" onClick={() => openNotification(notification)} className="text-xs font-medium text-accent-700 hover:text-accent-600">{notification.kind === 'approval_required' ? reviewing ? 'Hide review' : 'Review' : 'Open Event'}</button></div>
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
            <div className="px-4 py-12 text-center sm:px-6"><CheckCircle2 size={22} className="mx-auto text-accent-700" /><p className="mt-3 text-sm font-medium text-zinc-300">You are all caught up</p><p className="mt-1 text-xs text-zinc-500">Results, decisions, and meaningful changes will appear here.</p></div>
          )}
        </div>
      </Modal>
    </div>
  );
}
