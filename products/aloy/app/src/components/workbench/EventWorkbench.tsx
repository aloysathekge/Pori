import { useRef, type PointerEvent as ReactPointerEvent } from 'react';
import { Columns2, History, LayoutTemplate, Maximize2, Minimize2, X } from 'lucide-react';
import type { EventFile } from '@/api/events';
import type { StoredFileReference } from '@/hooks/useAttachments';
import { RunReplay } from '@/components/chat/RunReplay';
import { SurfaceFrame } from '@/components/surfaces/SurfaceFrame';
import type { SurfaceAloyHandoff } from '@/components/surfaces/surfaceBridge';
import { FileTypeIcon } from '@/components/files/FileVisual';
import { ArtifactViewer } from './ArtifactViewer';
import { StoredFileViewer } from './StoredFileViewer';

export type WorkbenchTab =
  | { id: 'surface'; kind: 'surface'; label: 'Surface' }
  | { id: string; kind: 'artifact'; label: string; path: string }
  | { id: string; kind: 'file'; label: string; file: EventFile }
  | { id: string; kind: 'replay'; label: string; runId: string };

// eslint-disable-next-line react-refresh/only-export-components -- shared with EventPage for the persistent pinned tab
export const SURFACE_TAB: WorkbenchTab = { id: 'surface', kind: 'surface', label: 'Surface' };

interface EventWorkbenchProps {
  eventId: string;
  eventTitle: string;
  conversationId: string;
  refreshKey?: string;
  tabs: WorkbenchTab[];
  activeTabId: string;
  onSelectTab: (tabId: string) => void;
  onCloseTab: (tabId: string) => void;
  onDismiss: () => void;
  onAskAloy: (reference: StoredFileReference) => void;
  showSurfaceAlongside: boolean;
  onToggleSurfaceAlongside: () => void;
  resourceRatio: number;
  onResourceRatioChange: (ratio: number) => void;
  focused: boolean;
  onToggleFocus: () => void;
  onSurfaceAloyHandoff: (handoff: SurfaceAloyHandoff) => void;
}

function TabIcon({ tab }: { tab: WorkbenchTab }) {
  if (tab.kind === 'surface') return <LayoutTemplate size={13} className="shrink-0" />;
  if (tab.kind === 'replay') return <History size={13} className="shrink-0" />;
  if (tab.kind === 'artifact') return <FileTypeIcon file={{ name: tab.path, kind: 'artifact' }} size={13} />;
  return <FileTypeIcon file={tab.file} size={13} />;
}

export function EventWorkbench({
  eventId,
  eventTitle,
  conversationId,
  refreshKey,
  tabs,
  activeTabId,
  onSelectTab,
  onCloseTab,
  onDismiss,
  onAskAloy,
  showSurfaceAlongside,
  onToggleSurfaceAlongside,
  resourceRatio,
  onResourceRatioChange,
  focused,
  onToggleFocus,
  onSurfaceAloyHandoff,
}: EventWorkbenchProps) {
  const splitRef = useRef<HTMLDivElement | null>(null);
  const activeTab = tabs.find((tab) => tab.id === activeTabId) ?? tabs[0] ?? SURFACE_TAB;
  const canShowSurfaceAlongside = activeTab.kind !== 'surface';

  function startResize(event: ReactPointerEvent<HTMLButtonElement>) {
    event.preventDefault();
    const split = splitRef.current;
    if (!split) return;
    const move = (pointer: PointerEvent) => {
      const bounds = split.getBoundingClientRect();
      const ratio = ((pointer.clientX - bounds.left) / bounds.width) * 100;
      onResourceRatioChange(Math.min(70, Math.max(30, ratio)));
    };
    const stop = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', stop);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', stop, { once: true });
  }

  function activeContent() {
    if (activeTab.kind === 'surface') {
      return <SurfaceFrame eventId={eventId} eventTitle={eventTitle} refreshKey={refreshKey} onAloyHandoff={onSurfaceAloyHandoff} />;
    }
    if (activeTab.kind === 'artifact') {
      return <ArtifactViewer conversationId={conversationId} path={activeTab.path} onAskAloy={onAskAloy} />;
    }
    if (activeTab.kind === 'file') {
      return <StoredFileViewer file={activeTab.file} onAskAloy={onAskAloy} />;
    }
    return <RunReplay runId={activeTab.runId} embedded />;
  }

  return (
    <section className="flex h-full min-h-0 min-w-0 flex-col overflow-hidden bg-zinc-950">
      <div className="flex h-11 shrink-0 items-center border-b border-zinc-800 bg-zinc-900/70">
        <div className="flex min-w-0 flex-1 self-stretch overflow-x-auto px-1.5 pt-1.5">
          {tabs.map((tab) => {
            const active = tab.id === activeTab.id;
            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => onSelectTab(tab.id)}
                className={`group flex min-w-0 max-w-48 shrink-0 items-center gap-1.5 rounded-t-lg border border-b-0 px-2.5 text-xs transition-colors ${active ? 'border-zinc-800 bg-zinc-950 text-zinc-200' : 'border-transparent text-zinc-500 hover:bg-zinc-800/70 hover:text-zinc-300'}`}
                title={tab.label}
              >
                <TabIcon tab={tab} />
                <span className="truncate">{tab.label}</span>
                {tab.kind !== 'surface' && (
                  <span
                    role="button"
                    tabIndex={0}
                    aria-label={`Close ${tab.label}`}
                    onClick={(event) => { event.stopPropagation(); onCloseTab(tab.id); }}
                    onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.stopPropagation(); onCloseTab(tab.id); } }}
                    className="ml-1 flex h-7 w-7 items-center justify-center rounded text-zinc-600 opacity-100 hover:bg-zinc-700 hover:text-zinc-200 sm:h-6 sm:w-6 md:opacity-0 md:group-hover:opacity-100 md:group-focus-visible:opacity-100"
                  >
                    <X size={11} />
                  </span>
                )}
              </button>
            );
          })}
        </div>
        {canShowSurfaceAlongside && (
          <button type="button" onClick={onToggleSurfaceAlongside} className={`mr-1 hidden rounded-md p-1.5 2xl:block ${showSurfaceAlongside ? 'bg-accent-600/10 text-accent-700' : 'text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200'}`} title={showSurfaceAlongside ? 'Hide Surface beside this tab' : 'Show Surface beside this tab'} aria-pressed={showSurfaceAlongside}>
            <Columns2 size={15} />
          </button>
        )}
        <button
          type="button"
          onClick={onToggleFocus}
          className="mr-1 flex h-10 w-10 shrink-0 items-center justify-center rounded-lg text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200"
          title={focused ? 'Exit Focus Mode' : 'Open Focus Mode'}
          aria-label={focused ? 'Exit Workbench Focus Mode' : 'Open Workbench Focus Mode'}
          aria-pressed={focused}
        >
          {focused ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
        </button>
        <button type="button" onClick={onDismiss} className="mr-1 flex h-10 w-10 shrink-0 items-center justify-center rounded-lg text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200 sm:mr-2" title="Close Workbench" aria-label="Close Workbench"><X size={16} /></button>
      </div>

      <div ref={splitRef} className="flex min-h-0 min-w-0 flex-1">
        {canShowSurfaceAlongside && showSurfaceAlongside && (
          <>
            <div className="hidden min-h-0 min-w-0 flex-none 2xl:block" style={{ flexBasis: `${resourceRatio}%` }}>
              <SurfaceFrame eventId={eventId} eventTitle={eventTitle} refreshKey={refreshKey} onAloyHandoff={onSurfaceAloyHandoff} />
            </div>
            <button type="button" onPointerDown={startResize} className="group relative hidden w-1 shrink-0 cursor-col-resize bg-zinc-800 hover:bg-accent-600 2xl:block" aria-label="Resize Surface and active Workbench tab" title="Drag to resize"><span className="absolute left-1/2 top-1/2 h-10 w-0.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-zinc-600 group-hover:bg-white/70" /></button>
          </>
        )}
        <div className="min-h-0 min-w-0 flex-1">{activeContent()}</div>
      </div>
    </section>
  );
}
