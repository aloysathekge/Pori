import { Spinner } from '@/components/ui/Spinner';
import { Badge } from '@/components/ui/Badge';
import type { PlanItem, PlanItemStatus, SSEToolEvent } from '@/types';

interface Props {
  status: string;
  activity: string;
  plan: PlanItem[];
  tools: SSEToolEvent[];
  step?: { step: number; max_steps: number };
}

const PLAN_MARK: Record<PlanItemStatus, string> = {
  completed: '✓',
  in_progress: '▸',
  pending: '○',
  cancelled: '✕',
};

const PLAN_CLASS: Record<PlanItemStatus, string> = {
  completed: 'text-emerald-700 line-through opacity-70',
  in_progress: 'text-accent-700 font-medium',
  pending: 'text-zinc-400',
  cancelled: 'text-zinc-500 line-through opacity-60',
};

export function StreamingIndicator({
  status,
  activity,
  plan,
  tools,
  step,
}: Props) {
  const headline = activity || status || 'Thinking…';
  const visiblePlan = plan.filter((p) => p.status !== 'cancelled');
  const recentTools = tools.slice(-6);

  return (
    <div className="flex gap-3">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-zinc-700">
        <Spinner className="h-4 w-4" />
      </div>
      <div className="w-full max-w-2xl space-y-3 rounded-2xl bg-zinc-800 px-4 py-3 text-sm">
        {/* Activity headline (model's next_goal) */}
        <div className="flex items-center gap-2 text-zinc-200">
          <Spinner className="h-3 w-3" />
          <span>{headline}</span>
          {step && (
            <span className="ml-auto shrink-0 text-xs text-zinc-500">
              step {step.step}/{step.max_steps}
            </span>
          )}
        </div>

        {/* Live plan checklist */}
        {visiblePlan.length > 0 && (
          <ul className="space-y-1 border-l border-zinc-700 pl-3 text-xs">
            {visiblePlan.map((item) => (
              <li key={item.id} className={PLAN_CLASS[item.status]}>
                <span className="mr-1.5">{PLAN_MARK[item.status]}</span>
                {item.content}
              </li>
            ))}
          </ul>
        )}

        {/* Tool activity log */}
        {recentTools.length > 0 && (
          <div className="space-y-1">
            {recentTools.map((t, i) => (
              <div
                key={`${t.step}-${t.tool}-${i}`}
                className="flex items-center gap-2 text-xs text-zinc-400"
              >
                <Badge color={t.success ? 'green' : 'red'}>{t.tool}</Badge>
                <span className="truncate text-zinc-500">{t.preview}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
