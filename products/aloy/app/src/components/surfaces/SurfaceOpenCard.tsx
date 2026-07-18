import { useEffect, useState } from 'react';
import { PanelRightOpen } from 'lucide-react';
import {
  getPublishedSurfaceRuntime,
  surfaceSeenKey,
  type SurfaceBuild,
} from '@/api/surfaces';

interface SurfaceOpenCardProps {
  eventId: string;
  eventTitle: string;
  refreshKey?: string;
  visible: boolean;
  onOpen: () => void;
}

export function SurfaceOpenCard({
  eventId,
  eventTitle,
  refreshKey,
  visible,
  onOpen,
}: SurfaceOpenCardProps) {
  const [build, setBuild] = useState<SurfaceBuild | null>(null);

  useEffect(() => {
    if (!visible) return;
    let cancelled = false;
    void getPublishedSurfaceRuntime(eventId)
      .then((runtime) => {
        if (cancelled) return;
        const latest = runtime.build;
        const seenBuild = window.localStorage.getItem(surfaceSeenKey(eventId));
        setBuild(
          runtime.published_build_id && latest?.id === runtime.published_build_id
            && latest.id !== seenBuild
            ? latest
            : null,
        );
      })
      .catch(() => {
        if (!cancelled) setBuild(null);
      });
    return () => {
      cancelled = true;
    };
  }, [eventId, refreshKey, visible]);

  if (!visible || !build) return null;

  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900 p-4 shadow-sm">
      <div className="flex items-center gap-4">
        <div className="min-w-0 flex-1">
          <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-accent-700">
            Surface ready
          </p>
          <p className="mt-1 font-display text-base font-semibold text-zinc-100">
            Open the {eventTitle} Surface
          </p>
          <p className="mt-1 text-sm leading-5 text-zinc-500">
            Aloy has a new visual workspace for this Event. Open it beside the conversation.
          </p>
        </div>
        <button
          type="button"
          onClick={() => {
            window.localStorage.setItem(surfaceSeenKey(eventId), build.id);
            setBuild(null);
            onOpen();
          }}
          className="flex shrink-0 items-center gap-2 rounded-xl border border-accent-600/25 bg-accent-600/10 px-3.5 py-2.5 text-sm font-semibold text-accent-700 transition hover:border-accent-600/40 hover:bg-accent-600/15 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-500"
        >
          <PanelRightOpen size={16} />
          Open Surface
        </button>
      </div>
    </div>
  );
}
