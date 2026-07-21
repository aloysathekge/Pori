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
    <div className="flex justify-start">
      <div className="w-full max-w-3xl space-y-3 py-1 text-sm">
        {/* Live activity headline */}
        <div className="flex items-center gap-2.5 text-zinc-300">
          <span className="relative flex h-2.5 w-2.5 shrink-0">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-accent-500 opacity-60" />
            <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-accent-600" />
          </span>
          <span className="animate-pulse font-medium">{headline}</span>
          {step && step.max_steps > 0 && (
            <span className="ml-auto shrink-0 text-xs text-zinc-500">
              step {step.step} of {step.max_steps}
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
