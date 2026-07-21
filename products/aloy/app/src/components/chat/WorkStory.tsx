import { useEffect, useMemo, useState } from 'react';
import {
  AlertCircle,
  Check,
  ChevronDown,
  ChevronRight,
  Circle,
  LoaderCircle,
  X,
} from 'lucide-react';
import { getRunTimeline } from '@/api/runEvents';
import type { PlanItem, RunTimelineEvent } from '@/types';

interface WorkStoryProps {
  runId?: string | null;
  entries?: RunTimelineEvent[];
  live?: boolean;
}

interface StoryAction {
  id: string;
  label: string;
  status: 'running' | 'succeeded' | 'failed';
  durationSeconds?: number;
}

function text(payload: Record<string, unknown>, key: string) {
  const value = payload[key];
  return typeof value === 'string' ? value : '';
}

function formatDuration(seconds: number) {
  if (seconds < 1) return '<1s';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.round(seconds % 60);
  return remainder ? `${minutes}m ${remainder}s` : `${minutes}m`;
}

function planFrom(entries: RunTimelineEvent[]): PlanItem[] {
  const latest = [...entries]
    .reverse()
    .find((entry) => entry.kind === 'plan_changed');
  const plan = latest?.public_payload.plan;
  if (!Array.isArray(plan)) return [];
  return plan.filter(
    (item): item is PlanItem =>
      !!item
      && typeof item === 'object'
      && typeof (item as PlanItem).id === 'string'
      && typeof (item as PlanItem).content === 'string'
      && ['pending', 'in_progress', 'completed', 'cancelled'].includes(
        (item as PlanItem).status,
      ),
  );
}

export function WorkStory({ runId, entries: suppliedEntries = [], live = false }: WorkStoryProps) {
  const [durableEntries, setDurableEntries] = useState<RunTimelineEvent[]>(
    suppliedEntries,
  );
  const [expanded, setExpanded] = useState(live);

  useEffect(() => {
    if (!runId || live) return;
    let cancelled = false;
    let retryTimer: ReturnType<typeof setTimeout> | undefined;

    const load = async (attempt: number) => {
      try {
        const page = await getRunTimeline(runId);
        if (cancelled) return;
        if (page.entries.length > 0) {
          setDurableEntries(page.entries);
          return;
        }
      } catch {
        // The final SSE frame can arrive just before Run persistence commits.
      }
      const optimisticFinal = suppliedEntries.length > 0;
      if (!cancelled && optimisticFinal && attempt < 4) {
        retryTimer = setTimeout(() => void load(attempt + 1), 500 * (attempt + 1));
      }
    };

    void load(0);
    return () => {
      cancelled = true;
      if (retryTimer) clearTimeout(retryTimer);
    };
  }, [live, runId, suppliedEntries.length]);

  const entries = live ? suppliedEntries : durableEntries;
  const story = useMemo(() => {
    const actions = new Map<string, StoryAction>();
    let activity = '';
    let failed = false;
    let cancelled = false;
    let completed = false;
    let needsAttention = false;

    for (const entry of entries) {
      const payload = entry.public_payload;
      if (entry.kind === 'activity_changed') {
        activity = text(payload, 'activity');
      } else if (entry.kind === 'action_started') {
        const id = text(payload, 'call_id') || entry.id;
        actions.set(id, {
          id,
          label: text(payload, 'label') || 'Working',
          status: 'running',
        });
      } else if (entry.kind === 'action_finished') {
        const id = text(payload, 'call_id') || entry.id;
        actions.set(id, {
          id,
          label: text(payload, 'label') || actions.get(id)?.label || 'Finished an action',
          status: payload.success === false ? 'failed' : 'succeeded',
          durationSeconds:
            typeof payload.duration_seconds === 'number'
              ? payload.duration_seconds
              : undefined,
        });
      } else if (entry.kind === 'attention_required') {
        needsAttention = true;
      } else if (entry.kind === 'run_finished') {
        completed = payload.completed !== false;
      } else if (entry.kind === 'run_failed') {
        failed = true;
      } else if (entry.kind === 'run_cancelled') {
        cancelled = true;
      }
    }

    const actionList = [...actions.values()];
    const durationFromActions = actionList.reduce(
      (total, action) => total + (action.durationSeconds ?? 0),
      0,
    );
    const firstAt = entries[0] ? new Date(entries[0].created_at).getTime() : 0;
    const lastAt = entries.at(-1) ? new Date(entries.at(-1)!.created_at).getTime() : 0;
    const elapsed = durationFromActions || Math.max(0, (lastAt - firstAt) / 1000);
    return {
      actions: actionList,
      activity,
      completed,
      failed,
      cancelled,
      needsAttention,
      elapsed,
      plan: planFrom(entries),
    };
  }, [entries]);

  if (entries.length === 0) return null;

  const finishedActions = story.actions.filter((action) => action.status !== 'running');
  const runningAction = story.actions.findLast((action) => action.status === 'running');
  const summary = story.failed
    ? `Work stopped · ${finishedActions.length} action${finishedActions.length === 1 ? '' : 's'}`
    : story.cancelled
      ? `Work stopped · ${finishedActions.length} action${finishedActions.length === 1 ? '' : 's'}`
    : story.completed
      ? `Worked for ${formatDuration(story.elapsed)} · ${finishedActions.length} action${finishedActions.length === 1 ? '' : 's'}`
      : story.needsAttention
        ? 'Waiting for you'
        : story.activity || runningAction?.label || 'Working on your request';

  return (
    <section className="mb-5 max-w-3xl text-sm text-zinc-400" aria-live={live ? 'polite' : 'off'}>
      <button
        type="button"
        onClick={() => setExpanded((value) => !value)}
        className="group/story flex w-full items-center gap-2 py-1 text-left transition-colors hover:text-zinc-200"
        aria-expanded={expanded}
      >
        {live && !story.completed && !story.failed && !story.cancelled ? (
          <LoaderCircle size={14} className="shrink-0 animate-spin text-accent-500" />
        ) : story.failed ? (
          <AlertCircle size={14} className="shrink-0 text-red-400" />
        ) : story.cancelled ? (
          <X size={14} className="shrink-0 text-zinc-400" />
        ) : (
          <Check size={14} className="shrink-0 text-emerald-500" />
        )}
        <span className="min-w-0 flex-1 truncate font-medium">{summary}</span>
        {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
      </button>

      {expanded && (
        <div className="ml-1.5 mt-2 space-y-3 border-l border-zinc-800 pl-4">
          {story.plan.length > 0 && (
            <ol className="space-y-1.5">
              {story.plan
                .filter((item) => item.status !== 'cancelled')
                .map((item) => (
                  <li key={item.id} className="flex items-start gap-2 text-xs leading-5">
                    {item.status === 'completed' ? (
                      <Check size={13} className="mt-1 shrink-0 text-emerald-500" />
                    ) : item.status === 'in_progress' ? (
                      <LoaderCircle size={13} className="mt-1 shrink-0 animate-spin text-accent-500" />
                    ) : (
                      <Circle size={11} className="mt-1.5 shrink-0 text-zinc-600" />
                    )}
                    <span className={item.status === 'completed' ? 'text-zinc-500 line-through' : ''}>
                      {item.content}
                    </span>
                  </li>
                ))}
            </ol>
          )}

          {story.actions.length > 0 && (
            <div className="space-y-1.5">
              {story.actions.slice(-8).map((action) => (
                <div key={action.id} className="flex items-start gap-2 text-xs leading-5">
                  {action.status === 'running' ? (
                    <LoaderCircle size={13} className="mt-1 shrink-0 animate-spin text-accent-500" />
                  ) : action.status === 'failed' ? (
                    <X size={13} className="mt-1 shrink-0 text-red-400" />
                  ) : (
                    <Check size={13} className="mt-1 shrink-0 text-emerald-500" />
                  )}
                  <span>{action.label}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </section>
  );
}
