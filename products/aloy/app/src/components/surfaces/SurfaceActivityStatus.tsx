import { AlertTriangle, CheckCircle2, LoaderCircle } from 'lucide-react';
import type { SurfaceActivity } from '@/api/surfaces';

interface SurfaceActivityStatusProps {
  activity: SurfaceActivity;
  compact?: boolean;
}

const STEPS = ['Generate', 'Validate', 'Build', 'Publish'] as const;

function currentStep(activity: SurfaceActivity) {
  if (activity.status === 'completed') return STEPS.length;
  if (activity.stage === 'validating_candidate') return 1;
  if (activity.stage === 'building_bundle' || activity.stage === 'inspecting_preview') return 2;
  if (activity.stage === 'publishing_surface') return 3;
  return 0;
}

function duration(seconds: number) {
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  return `${minutes}m ${remainder.toString().padStart(2, '0')}s`;
}

export function SurfaceActivityStatus({
  activity,
  compact = false,
}: SurfaceActivityStatusProps) {
  const active = activity.active;
  const ready = activity.status === 'completed';
  const failed = ['failed', 'cancelled', 'overdue'].includes(activity.status);
  const step = currentStep(activity);
  const retry = activity.attempt_count > 1
    ? `Builder attempt ${activity.attempt_count} of ${activity.max_attempts}`
    : activity.submission > 1
      ? `Candidate ${activity.submission} of 2`
      : null;

  return (
    <div className={compact ? '' : 'mx-auto w-full max-w-md'} role="status" aria-live="polite">
      <div className="flex items-start gap-3">
        <div className={`mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-xl border ${
          failed
            ? 'border-red-500/20 bg-red-500/10 text-red-400'
            : ready
              ? 'border-emerald-500/20 bg-emerald-500/10 text-emerald-400'
              : 'border-accent-600/20 bg-accent-600/10 text-accent-600'
        }`}>
          {active && <LoaderCircle size={16} className="animate-spin" />}
          {ready && <CheckCircle2 size={16} />}
          {failed && <AlertTriangle size={16} />}
        </div>
        <div className="min-w-0 flex-1">
          <p className={`${compact ? 'text-sm' : 'text-base'} font-semibold text-zinc-200`}>
            {activity.message}
          </p>
          <p className="mt-1 text-xs leading-5 text-zinc-500">
            {active
              ? `Aloy is working in the background · ${duration(activity.elapsed_seconds)}`
              : ready
                ? 'The new visual workspace has been published.'
                : 'Your conversation and last working Surface remain available.'}
          </p>
          {retry && <p className="mt-1 text-[10px] font-medium text-amber-500">{retry}</p>}
        </div>
      </div>

      <div className={`${compact ? 'mt-3' : 'mt-5'} grid grid-cols-4 gap-1.5`} aria-label="Surface build stages">
        {STEPS.map((label, index) => {
          const complete = ready || index < step;
          const current = active && index === step;
          return (
            <div key={label} className="min-w-0">
              <div className={`h-1 rounded-full ${
                complete
                  ? 'bg-emerald-500'
                  : current
                    ? 'animate-pulse bg-accent-600'
                    : failed && index === step
                      ? 'bg-red-500'
                      : 'bg-zinc-800'
              }`} />
              {!compact && (
                <p className={`mt-1.5 truncate text-[10px] ${current ? 'font-medium text-zinc-300' : 'text-zinc-600'}`}>
                  {label}
                </p>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
