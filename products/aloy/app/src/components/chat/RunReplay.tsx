import { useCallback, useEffect, useState } from 'react';
import {
  Brain,
  Check,
  MessageSquareText,
  Pause,
  Play,
  RotateCcw,
  Wrench,
  X,
} from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Spinner } from '@/components/ui/Spinner';
import { getRunEvents, type RunEvent } from '@/api/runEvents';

const STEP_MS = 500;

function eventText(e: RunEvent): string {
  const p = e.payload || {};
  if (typeof p.text === 'string') return p.text;
  if (typeof p.name === 'string') return p.name;
  return '';
}

function Row({ event }: { event: RunEvent }) {
  const text = eventText(event);
  switch (event.type) {
    case 'thinking':
      return (
        <div className="flex gap-2 text-sm text-zinc-500">
          <Brain size={15} className="mt-0.5 shrink-0 text-zinc-400" />
          <p className="whitespace-pre-wrap italic">{text}</p>
        </div>
      );
    case 'text':
      return (
        <div className="flex gap-2 text-sm text-zinc-200">
          <MessageSquareText size={15} className="mt-0.5 shrink-0 text-accent-600" />
          <p className="whitespace-pre-wrap">{text}</p>
        </div>
      );
    case 'tool_call_start':
      return (
        <div className="flex items-center gap-2 text-sm text-zinc-300">
          <Wrench size={15} className="shrink-0 text-zinc-400" />
          <span>
            Ran <span className="font-mono text-zinc-100">{text}</span>
          </span>
        </div>
      );
    case 'tool_call_end': {
      const ok = event.payload?.success !== false;
      return (
        <div className="flex items-center gap-2 pl-6 text-xs text-zinc-500">
          <Check
            size={13}
            className={ok ? 'text-accent-600' : 'text-red-600'}
          />
          <span>{ok ? 'completed' : 'failed'}</span>
        </div>
      );
    }
    case 'step_start':
      return (
        <div className="flex items-center gap-2 pt-1 text-xs font-medium uppercase tracking-wide text-zinc-400">
          <span className="h-px flex-1 bg-zinc-800" />
          Step {event.step}
          <span className="h-px flex-1 bg-zinc-800" />
        </div>
      );
    case 'llm_retry':
      return (
        <p className="pl-6 text-xs text-amber-700">Provider retry…</p>
      );
    case 'run_end':
      return (
        <p className="pt-1 text-xs font-medium text-accent-600">
          Run complete
        </p>
      );
    case 'truncated':
      return (
        <p className="text-xs text-zinc-500">
          {String(event.payload?.reason || 'log truncated')}
        </p>
      );
    default:
      return null;
  }
}

export function RunReplay({
  runId,
  onClose,
}: {
  runId: string;
  onClose: () => void;
}) {
  const [events, setEvents] = useState<RunEvent[] | null>(null);
  const [error, setError] = useState('');
  const [cursor, setCursor] = useState(0); // events revealed
  const [playing, setPlaying] = useState(false);

  const load = useCallback(async () => {
    try {
      const log = await getRunEvents(runId);
      setEvents(log.events);
      setCursor(log.events.length); // show it all by default; Play re-runs it
    } catch (e) {
      setError(e instanceof Error ? e.message : 'No replay available');
    }
  }, [runId]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    load();
  }, [load]);

  useEffect(() => {
    // Advance the cursor on a timer while playing; stops naturally at the end
    // (no timer scheduled). Only async setState — no synchronous effect write.
    if (!playing || !events || cursor >= events.length) return;
    const id = setTimeout(() => setCursor((c) => c + 1), STEP_MS);
    return () => clearTimeout(id);
  }, [playing, cursor, events]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  const total = events?.length ?? 0;
  const shown = events?.slice(0, cursor) ?? [];

  function restart() {
    setCursor(0);
    setPlaying(true);
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="fixed inset-0 bg-black/30" onClick={onClose} />
      <div className="relative z-50 flex max-h-[85vh] w-full max-w-2xl flex-col rounded-xl border border-zinc-800 bg-zinc-900 shadow-2xl">
        <div className="flex items-center justify-between border-b border-zinc-800 px-5 py-3">
          <div>
            <h2 className="text-base font-semibold text-zinc-100">Replay</h2>
            <p className="text-xs text-zinc-500">
              What the agent did, step by step
            </p>
          </div>
          <button
            aria-label="Close replay"
            onClick={onClose}
            className="rounded-lg p-1.5 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100"
          >
            <X size={18} />
          </button>
        </div>

        <div className="flex-1 space-y-3 overflow-y-auto px-5 py-4">
          {error ? (
            <p className="py-10 text-center text-sm text-zinc-500">{error}</p>
          ) : !events ? (
            <div className="flex justify-center py-10">
              <Spinner className="h-6 w-6" />
            </div>
          ) : total === 0 ? (
            <p className="py-10 text-center text-sm text-zinc-500">
              No events were recorded for this run.
            </p>
          ) : (
            shown.map((e, i) => <Row key={i} event={e} />)
          )}
        </div>

        {events && total > 0 && (
          <div className="flex items-center gap-3 border-t border-zinc-800 px-5 py-3">
            <Button
              size="sm"
              variant="secondary"
              onClick={() =>
                cursor >= total ? restart() : setPlaying((p) => !p)
              }
            >
              {cursor >= total ? (
                <>
                  <RotateCcw size={14} /> Replay
                </>
              ) : playing ? (
                <>
                  <Pause size={14} /> Pause
                </>
              ) : (
                <>
                  <Play size={14} /> Play
                </>
              )}
            </Button>
            <input
              type="range"
              min={0}
              max={total}
              value={cursor}
              aria-label="Scrub replay"
              onChange={(e) => {
                setPlaying(false);
                setCursor(Number(e.target.value));
              }}
              className="h-1 flex-1 cursor-pointer accent-accent-600"
            />
            <span className="w-14 text-right text-xs tabular-nums text-zinc-500">
              {Math.min(cursor, total)}/{total}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
