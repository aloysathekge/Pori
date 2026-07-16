import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { RefreshCw, ShieldCheck, WifiOff } from 'lucide-react';
import {
  getSurfaceRuntimeDocument,
  listSurfaceBuilds,
  surfaceSeenKey,
  type SurfaceBuild,
} from '@/api/surfaces';
import { Spinner } from '@/components/ui/Spinner';
import { SurfaceBridgeHost } from './surfaceBridge';

interface SurfaceFrameProps {
  eventId: string;
  eventTitle: string;
  refreshKey?: string;
}

type FrameState =
  | { kind: 'loading' }
  | { kind: 'empty'; latest: SurfaceBuild | null }
  | { kind: 'error'; message: string; build: SurfaceBuild | null }
  | { kind: 'ready'; url: string; build: SurfaceBuild };

type RuntimeState = {
  status: 'idle' | 'connecting' | 'healthy' | 'reconnecting' | 'degraded';
  message?: string;
  attempt?: number;
};

const RECONNECT_DELAYS = [1_000, 2_000, 4_000] as const;

export function SurfaceFrame({ eventId, eventTitle, refreshKey }: SurfaceFrameProps) {
  const [state, setState] = useState<FrameState>({ kind: 'loading' });
  const [runtime, setRuntime] = useState<RuntimeState>({ status: 'idle' });
  const [reload, setReload] = useState(0);
  const objectUrl = useRef<string | null>(null);
  const requestId = useRef(0);
  const iframeRef = useRef<HTMLIFrameElement | null>(null);
  const bridgeRef = useRef<SurfaceBridgeHost | null>(null);
  const currentBuildId = useRef<string | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectAttempt = useRef(0);

  const clearReconnect = useCallback(() => {
    if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
    reconnectTimer.current = null;
  }, []);

  const load = useCallback(async () => {
    const currentRequest = ++requestId.current;
    clearReconnect();
    reconnectAttempt.current = 0;
    bridgeRef.current?.disconnect(false);
    bridgeRef.current = null;
    currentBuildId.current = null;
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    objectUrl.current = null;
    setRuntime({ status: 'idle' });
    setState({ kind: 'loading' });
    try {
      const builds = await listSurfaceBuilds(eventId);
      if (currentRequest !== requestId.current) return;
      const build = builds.find(
        (candidate) => candidate.status === 'succeeded' && candidate.bundle_available,
      );
      if (!build) {
        setState({ kind: 'empty', latest: builds[0] ?? null });
        return;
      }
      const document = await getSurfaceRuntimeDocument(eventId, build.id);
      const url = URL.createObjectURL(new Blob([document], { type: 'text/html' }));
      if (currentRequest !== requestId.current) {
        URL.revokeObjectURL(url);
        return;
      }
      objectUrl.current = url;
      currentBuildId.current = build.id;
      window.localStorage.setItem(surfaceSeenKey(eventId), build.id);
      setState({ kind: 'ready', url, build });
    } catch (cause) {
      if (currentRequest !== requestId.current) return;
      setState({
        kind: 'error',
        message: cause instanceof Error ? cause.message : String(cause),
        build: null,
      });
    }
  }, [clearReconnect, eventId]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- build identity drives an authenticated runtime reload
    void load();
  }, [load, reload]);

  useEffect(() => {
    if (!refreshKey || !currentBuildId.current) return;
    let cancelled = false;
    const refresh = async () => {
      try {
        const builds = await listSurfaceBuilds(eventId);
        if (cancelled) return;
        const build = builds.find(
          (candidate) => candidate.status === 'succeeded' && candidate.bundle_available,
        );
        if (build && build.id !== currentBuildId.current) {
          await load();
          return;
        }
        await bridgeRef.current?.refresh();
      } catch {
        // The existing last-good Surface stays mounted; the live stream will
        // provide another invalidation and manual reload remains available.
      }
    };
    void refresh();
    return () => { cancelled = true; };
  }, [eventId, load, refreshKey]);

  useEffect(() => {
    return () => {
      requestId.current += 1;
      clearReconnect();
      bridgeRef.current?.disconnect(false);
      if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    };
  }, [clearReconnect]);

  function scheduleReconnect(build: SurfaceBuild, message?: string) {
    if (reconnectTimer.current) return;
    const index = reconnectAttempt.current;
    if (index >= RECONNECT_DELAYS.length) {
      setRuntime({ status: 'degraded', message });
      return;
    }
    const attempt = index + 1;
    reconnectAttempt.current = attempt;
    setRuntime({ status: 'reconnecting', message, attempt });
    reconnectTimer.current = setTimeout(() => {
      reconnectTimer.current = null;
      void connectBridge(build, attempt);
    }, RECONNECT_DELAYS[index]);
  }

  async function connectBridge(build: SurfaceBuild, attempt = 0) {
    const frame = iframeRef.current;
    if (!frame) return;
    clearReconnect();
    bridgeRef.current?.disconnect(false);
    const bridge = new SurfaceBridgeHost(eventId, build.id, {
      onStatus(update) {
        if (bridgeRef.current !== bridge) return;
        if (update.status === 'healthy') {
          reconnectAttempt.current = 0;
          setRuntime({ status: 'healthy' });
        } else if (update.status === 'connecting') {
          setRuntime({
            status: attempt > 0 ? 'reconnecting' : 'connecting',
            attempt: attempt || undefined,
          });
        } else if (update.status === 'degraded') {
          scheduleReconnect(build, update.message);
        }
      },
    });
    bridgeRef.current = bridge;
    try {
      await bridge.connect(frame);
    } catch (cause) {
      if (bridgeRef.current !== bridge) return;
      scheduleReconnect(
        build,
        cause instanceof Error ? cause.message : 'Surface bridge unavailable',
      );
    }
  }

  function reconnectNow() {
    if (state.kind !== 'ready') return;
    clearReconnect();
    reconnectAttempt.current = 0;
    void connectBridge(state.build);
  }

  const buildLabel = useMemo(() => {
    if (state.kind !== 'ready') return null;
    return state.build.bundle_sha256?.slice(0, 8) ?? state.build.id.slice(0, 8);
  }, [state]);

  return (
    <section className="flex h-full min-h-0 flex-col overflow-hidden bg-zinc-950">
      <div className="flex h-11 shrink-0 items-center justify-between border-b border-zinc-800 px-3">
        <div className="flex min-w-0 items-center gap-2">
          <span className="text-xs font-semibold text-zinc-200">Surface</span>
          <span className="rounded-full border border-amber-500/20 bg-amber-500/10 px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wider text-amber-500">
            Preview
          </span>
          {buildLabel && (
            <span className="truncate font-mono text-[10px] text-zinc-600">{buildLabel}</span>
          )}
          {state.kind === 'ready' && (
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                runtime.status === 'healthy'
                  ? 'bg-emerald-500'
                  : runtime.status === 'degraded'
                    ? 'bg-red-500'
                    : 'animate-pulse bg-amber-500'
              }`}
              title={`Surface runtime: ${runtime.status}`}
            />
          )}
        </div>
        <div className="flex items-center gap-2">
          <span className="hidden items-center gap-1 text-[10px] text-zinc-600 sm:flex" title="Generated code runs without host access or network access">
            <ShieldCheck size={12} /> Isolated
          </span>
          <button
            type="button"
            onClick={() => setReload((value) => value + 1)}
            className="rounded-md p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200"
            aria-label="Reload Surface preview"
            title="Reload Surface preview"
          >
            <RefreshCw size={14} />
          </button>
        </div>
      </div>

      <div className="relative min-h-0 flex-1">
        {state.kind === 'loading' && (
          <div className="flex h-full items-center justify-center"><Spinner className="h-6 w-6" /></div>
        )}
        {state.kind === 'empty' && (
          <div className="flex h-full items-center justify-center px-8 text-center">
            <div className="max-w-sm">
              <div className="mx-auto mb-4 h-10 w-10 rounded-2xl border border-zinc-800 bg-zinc-900 shadow-inner" />
              <h2 className="font-display text-base font-semibold text-zinc-200">
                {state.latest?.status === 'running' || state.latest?.status === 'pending'
                  ? 'Aloy is building this Surface'
                  : `No Surface for ${eventTitle} yet`}
              </h2>
              <p className="mt-2 text-sm leading-6 text-zinc-500">
                {state.latest
                  ? `The latest build is ${state.latest.status}. The last successful preview will appear here automatically.`
                  : 'When Aloy creates and successfully builds a useful interface for this Event, it will appear here.'}
              </p>
            </div>
          </div>
        )}
        {state.kind === 'error' && (
          <div className="flex h-full items-center justify-center px-8 text-center">
            <div className="max-w-sm">
              <h2 className="font-display text-base font-semibold text-zinc-200">Surface preview unavailable</h2>
              <p className="mt-2 text-sm leading-6 text-zinc-500">{state.message}</p>
              <button type="button" onClick={() => setReload((value) => value + 1)} className="mt-4 text-sm font-medium text-accent-700 hover:text-accent-600">Try again</button>
            </div>
          </div>
        )}
        {state.kind === 'ready' && (
          <>
            <iframe
              key={state.url}
              ref={iframeRef}
              src={state.url}
              onLoad={() => void connectBridge(state.build)}
              sandbox="allow-scripts"
              referrerPolicy="no-referrer"
              title={`${eventTitle} Surface preview`}
              className="h-full w-full border-0 bg-white"
            />
            {(runtime.status === 'reconnecting' || runtime.status === 'degraded') && (
              <div
                className="absolute inset-x-3 bottom-3 flex items-center gap-3 rounded-xl border border-zinc-700/80 bg-zinc-950/95 px-3 py-2.5 shadow-2xl backdrop-blur"
                role="status"
                aria-live="polite"
              >
                <WifiOff size={15} className="shrink-0 text-amber-500" />
                <div className="min-w-0 flex-1">
                  <p className="text-xs font-medium text-zinc-200">
                    {runtime.status === 'reconnecting'
                      ? `Reconnecting to Aloy${runtime.attempt ? ` · attempt ${runtime.attempt}` : ''}`
                      : 'This Surface is temporarily offline'}
                  </p>
                  {runtime.message && (
                    <p className="mt-0.5 truncate text-[10px] text-zinc-500">{runtime.message}</p>
                  )}
                </div>
                {runtime.status === 'degraded' && (
                  <button
                    type="button"
                    onClick={reconnectNow}
                    className="shrink-0 rounded-md border border-zinc-700 px-2.5 py-1 text-xs font-medium text-zinc-200 hover:bg-zinc-800"
                  >
                    Reconnect
                  </button>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </section>
  );
}
