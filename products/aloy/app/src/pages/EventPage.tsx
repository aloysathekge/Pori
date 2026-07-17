import { useCallback, useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react';
import {
  Activity,
  BadgeCheck,
  CheckCircle2,
  Circle,
  FileText,
  ListTodo,
  PanelRightClose,
  PanelRightOpen,
  Play,
  Plus,
  RotateCcw,
  Send,
  ShieldCheck,
  Square,
  Trash2,
  Wifi,
  WifiOff,
  X,
} from 'lucide-react';
import { Link, useParams } from 'react-router-dom';
import { getConversation, getConversationMessages } from '@/api/conversations';
import {
  createEventTask,
  decideEventProposal,
  deleteEventTask,
  getEventSurface,
  getEventTrail,
  resumeEventTask,
  retryEventTask,
  stopEventTask,
  streamEventChanges,
  updateEventTask,
  workOnEventTask,
  type EventSurfaceResponse,
  type EventFile,
  type EventTask,
  type EventTrailEntry,
} from '@/api/events';
import { Composer } from '@/components/chat/Composer';
import { MessageList } from '@/components/chat/MessageList';
import { ProposalCard } from '@/components/events/ProposalCard';
import { EventCover } from '@/components/events/EventCover';
import { SurfaceOpenCard } from '@/components/surfaces/SurfaceOpenCard';
import { EventWorkbench, SURFACE_TAB, type WorkbenchTab } from '@/components/workbench/EventWorkbench';
import { Button } from '@/components/ui/Button';
import { Spinner } from '@/components/ui/Spinner';
import { useAttachments, type StoredFileReference } from '@/hooks/useAttachments';
import { useStreamingRun } from '@/hooks/useStreamingRun';
import type { MessageResponse } from '@/types';

type ContextTab = 'tasks' | 'approvals' | 'receipts' | 'files' | 'trail';
type WorkspaceMode = 'conversation' | 'split' | 'workbench';

const INPUT =
  'w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 placeholder-zinc-500 focus:border-accent-500 focus:outline-none';

function when(value: string) {
  return new Date(value).toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
}

function taskStatusLabel(status: EventTask['status']) {
  return status.replaceAll('_', ' ');
}

function taskCanToggle(status: EventTask['status']) {
  return status === 'open' || status === 'done';
}

export function EventPage() {
  const { eventId = '' } = useParams();
  return <EventPageWorkspace key={eventId} eventId={eventId} />;
}

function EventPageWorkspace({ eventId }: { eventId: string }) {
  const [data, setData] = useState<EventSurfaceResponse | null>(null);
  const [messages, setMessages] = useState<MessageResponse[]>([]);
  const [messageCursor, setMessageCursor] = useState<string | null>(null);
  const [loadingOlderMessages, setLoadingOlderMessages] = useState(false);
  const [trailEntries, setTrailEntries] = useState<EventTrailEntry[]>([]);
  const [trailCursor, setTrailCursor] = useState<string | null | undefined>(undefined);
  const [loadingOlderTrail, setLoadingOlderTrail] = useState(false);
  const [liveStatus, setLiveStatus] = useState<'connecting' | 'live' | 'reconnecting' | 'stale' | 'offline'>('connecting');
  const [input, setInput] = useState('');
  const [taskTitle, setTaskTitle] = useState('');
  const [loadingConversation, setLoadingConversation] = useState(true);
  const [contextOpen, setContextOpen] = useState(() => window.localStorage.getItem(`aloy:event:${eventId}:context-open`) !== 'false');
  const [contextTab, setContextTab] = useState<ContextTab>('tasks');
  const [workspaceMode, setWorkspaceMode] = useState<WorkspaceMode>(() => {
    const stored = window.localStorage.getItem(`aloy:event:${eventId}:workspace-mode`);
    if (stored === 'surface') return 'workbench';
    return stored === 'split' || stored === 'workbench' ? stored : 'conversation';
  });
  const [splitRatio, setSplitRatio] = useState(() => {
    const stored = Number(window.localStorage.getItem(`aloy:event:${eventId}:split-ratio`));
    return Number.isFinite(stored) && stored >= 30 && stored <= 70 ? stored : 50;
  });
  const [workbenchTabs, setWorkbenchTabs] = useState<WorkbenchTab[]>(() => {
    try {
      const stored = JSON.parse(window.localStorage.getItem(`aloy:event:${eventId}:workbench-tabs`) || '[]') as WorkbenchTab[];
      return [SURFACE_TAB, ...stored.filter((tab) => tab.id !== SURFACE_TAB.id)];
    } catch {
      return [SURFACE_TAB];
    }
  });
  const [activeWorkbenchTabId, setActiveWorkbenchTabId] = useState(() => window.localStorage.getItem(`aloy:event:${eventId}:workbench-active`) || SURFACE_TAB.id);
  const [showSurfaceAlongside, setShowSurfaceAlongside] = useState(() => window.localStorage.getItem(`aloy:event:${eventId}:surface-alongside`) === 'true');
  const [resourceRatio, setResourceRatio] = useState(() => {
    const stored = Number(window.localStorage.getItem(`aloy:event:${eventId}:resource-ratio`));
    return Number.isFinite(stored) && stored >= 30 && stored <= 70 ? stored : 50;
  });
  const [error, setError] = useState('');
  const [taskActionId, setTaskActionId] = useState<string | null>(null);
  const [resumeTaskId, setResumeTaskId] = useState<string | null>(null);
  const [resumeResponse, setResumeResponse] = useState('');
  const previousSending = useRef(false);
  const trailEventId = useRef<string | null>(null);
  const workspaceRef = useRef<HTMLDivElement | null>(null);

  const conversationId = data?.event.conversation_id ?? null;

  useEffect(() => {
    window.localStorage.setItem(`aloy:event:${eventId}:workspace-mode`, workspaceMode);
  }, [eventId, workspaceMode]);

  useEffect(() => {
    window.localStorage.setItem(`aloy:event:${eventId}:split-ratio`, String(splitRatio));
  }, [eventId, splitRatio]);

  useEffect(() => {
    window.localStorage.setItem(`aloy:event:${eventId}:context-open`, String(contextOpen));
    window.localStorage.setItem(`aloy:event:${eventId}:workbench-tabs`, JSON.stringify(workbenchTabs));
    window.localStorage.setItem(`aloy:event:${eventId}:workbench-active`, activeWorkbenchTabId);
    window.localStorage.setItem(`aloy:event:${eventId}:surface-alongside`, String(showSurfaceAlongside));
    window.localStorage.setItem(`aloy:event:${eventId}:resource-ratio`, String(resourceRatio));
  }, [activeWorkbenchTabId, contextOpen, eventId, resourceRatio, showSurfaceAlongside, workbenchTabs]);

  function startResize(event: ReactPointerEvent<HTMLButtonElement>) {
    event.preventDefault();
    const workspace = workspaceRef.current;
    if (!workspace) return;
    const move = (pointer: PointerEvent) => {
      const bounds = workspace.getBoundingClientRect();
      const ratio = ((pointer.clientX - bounds.left) / bounds.width) * 100;
      setSplitRatio(Math.min(70, Math.max(30, ratio)));
    };
    const stop = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', stop);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', stop, { once: true });
  }

  function openWorkbenchTab(tab: WorkbenchTab) {
    setWorkbenchTabs((current) => current.some((item) => item.id === tab.id) ? current : [...current, tab]);
    setActiveWorkbenchTabId(tab.id);
    setWorkspaceMode(window.matchMedia('(min-width: 768px)').matches ? 'split' : 'workbench');
  }

  function openSurface() {
    openWorkbenchTab(SURFACE_TAB);
  }

  function openArtifact(path: string) {
    openWorkbenchTab({
      id: `artifact:${path}`,
      kind: 'artifact',
      label: path.split(/[/\\]/).pop() || path,
      path,
    });
  }

  function closeWorkbenchTab(tabId: string) {
    if (tabId === SURFACE_TAB.id) return;
    setWorkbenchTabs((current) => {
      const index = current.findIndex((tab) => tab.id === tabId);
      const next = current.filter((tab) => tab.id !== tabId);
      if (activeWorkbenchTabId === tabId) {
        setActiveWorkbenchTabId(next[Math.max(0, index - 1)]?.id ?? SURFACE_TAB.id);
      }
      return next;
    });
  }

  const loadSurface = useCallback(async () => {
    if (!eventId) return;
    try {
      const next = await getEventSurface(eventId);
      setData(next);
      const activity = next.surface.sections.find((section) => section.kind === 'activity');
      if (activity?.kind === 'activity') {
        if (trailEventId.current !== eventId) {
          trailEventId.current = eventId;
          setTrailEntries(activity.entries);
          setTrailCursor(activity.next_cursor);
        } else {
          setTrailEntries((current) => {
            const merged = new Map(current.map((entry) => [entry.id, entry]));
            for (const entry of activity.entries) merged.set(entry.id, entry);
            return [...merged.values()].sort((a, b) =>
              b.created_at.localeCompare(a.created_at) || b.id.localeCompare(a.id),
            );
          });
          setTrailCursor((current) => current === undefined ? activity.next_cursor : current);
        }
      }
      setError('');
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }, [eventId]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- route-driven data load
    void loadSurface();
  }, [loadSurface]);

  const {
    pendingImages,
    pendingFiles,
    addAttachments,
    removeImage,
    removeFile,
    attachStoredFile,
    resetAttachments,
    uploadsInFlight,
    attachmentsFull,
  } = useAttachments(conversationId);

  const {
    sending,
    streaming,
    streamStatus,
    streamActivity,
    streamPlan,
    streamTools,
    streamStep,
    streamText,
    clarify,
    approval,
    resetStreamUi,
    abortStream,
    dispatchSend,
    tryReattach,
    resend,
    continueRun,
    stopRun,
    answerClarify,
    answerApproval,
  } = useStreamingRun({
    activeId: conversationId,
    setMessages,
    onConversationsRefresh: async () => undefined,
  });

  useEffect(() => {
    if (!conversationId) return;
    abortStream();
    resetStreamUi();
    // eslint-disable-next-line react-hooks/set-state-in-effect -- reset on canonical conversation change
    setLoadingConversation(true);
    let cancelled = false;
    getConversation(conversationId)
      .then((conversation) => {
        if (cancelled) return;
        setMessages(conversation.messages);
        setMessageCursor(conversation.messages_next_cursor);
        void tryReattach(conversationId);
      })
      .catch((cause) => {
        if (!cancelled) setError(cause instanceof Error ? cause.message : String(cause));
      })
      .finally(() => {
        if (!cancelled) setLoadingConversation(false);
      });
    return () => {
      cancelled = true;
      abortStream();
    };
    // Reattach is intentionally tied only to the canonical conversation id.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conversationId, abortStream, resetStreamUi]);

  useEffect(() => {
    if (previousSending.current && !sending) void loadSurface();
    previousSending.current = sending;
  }, [sending, loadSurface]);

  useEffect(() => {
    const controller = new AbortController();
    let cursor: string | null = null;
    let stopped = false;
    let attempts = 0;
    let lastFrame = Date.now();
    let refreshTimer: number | undefined;

    const refresh = (targetConversationId: string | null) => {
      window.clearTimeout(refreshTimer);
      refreshTimer = window.setTimeout(() => {
        void loadSurface();
        if (targetConversationId && targetConversationId === conversationId) {
          void getConversation(targetConversationId).then((conversation) => {
            setMessages(conversation.messages);
            setMessageCursor(conversation.messages_next_cursor);
          });
        }
      }, 120);
    };

    const follow = async () => {
      while (!stopped) {
        setLiveStatus(attempts ? 'reconnecting' : 'connecting');
        try {
          await streamEventChanges(
            eventId,
            cursor,
            {
              onCursor: (next) => {
                cursor = next;
                attempts = 0;
                lastFrame = Date.now();
                setLiveStatus('live');
              },
              onChange: (change) => {
                lastFrame = Date.now();
                setLiveStatus('live');
                setTrailEntries((current) => [
                  change.entry,
                  ...current.filter((entry) => entry.id !== change.entry.id),
                ]);
                refresh(change.conversation_id);
              },
              onHeartbeat: () => {
                lastFrame = Date.now();
                setLiveStatus('live');
              },
            },
            controller.signal,
          );
          if (!stopped) throw new Error('Event stream closed');
        } catch {
          if (stopped || controller.signal.aborted) return;
          attempts += 1;
          setLiveStatus(navigator.onLine ? 'reconnecting' : 'offline');
          await new Promise((resolve) =>
            window.setTimeout(resolve, Math.min(1000 * 2 ** attempts, 10_000)),
          );
        }
      }
    };
    void follow();
    const staleCheck = window.setInterval(() => {
      if (!navigator.onLine) setLiveStatus('offline');
      else if (Date.now() - lastFrame > 35_000) setLiveStatus('stale');
    }, 5_000);
    return () => {
      stopped = true;
      controller.abort();
      window.clearInterval(staleCheck);
      window.clearTimeout(refreshTimer);
    };
  }, [conversationId, eventId, loadSurface]);

  async function loadOlderMessages() {
    if (!conversationId || !messageCursor || loadingOlderMessages) return;
    setLoadingOlderMessages(true);
    try {
      const page = await getConversationMessages(conversationId, messageCursor);
      setMessages((current) => [...page.messages, ...current]);
      setMessageCursor(page.next_cursor);
    } finally {
      setLoadingOlderMessages(false);
    }
  }

  async function loadOlderTrail() {
    if (!trailCursor || loadingOlderTrail) return;
    setLoadingOlderTrail(true);
    try {
      const page = await getEventTrail(eventId, trailCursor);
      setTrailEntries((current) => [
        ...current,
        ...page.entries.filter(
          (entry) => !current.some((existing) => existing.id === entry.id),
        ),
      ]);
      setTrailCursor(page.next_cursor);
    } finally {
      setLoadingOlderTrail(false);
    }
  }

  async function handleSend() {
    if (!conversationId || uploadsInFlight) return;
    if (!input.trim() && pendingImages.length === 0 && pendingFiles.length === 0) return;
    const content = input.trim() || 'See the attached content.';
    if (clarify) {
      setInput('');
      answerClarify(content);
      return;
    }
    if (sending) return;
    const images = pendingImages;
    const files = pendingFiles;
    resetAttachments();
    setInput('');
    await dispatchSend(content, images, files);
  }

  async function addTask() {
    if (!taskTitle.trim()) return;
    try {
      await createEventTask(eventId, taskTitle.trim());
      setTaskTitle('');
      await loadSurface();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }

  async function toggleTask(taskId: string, status: EventTask['status']) {
    await updateEventTask(eventId, taskId, { status: status === 'done' ? 'open' : 'done' });
    await loadSurface();
  }

  async function removeTask(taskId: string) {
    await deleteEventTask(eventId, taskId);
    await loadSurface();
  }

  async function runTaskControl(
    task: EventTask,
    control: 'work' | 'stop' | 'retry' | 'resume',
    response?: string,
  ) {
    setTaskActionId(task.id);
    try {
      if (control === 'work') await workOnEventTask(eventId, task.id);
      if (control === 'stop') await stopEventTask(eventId, task.id);
      if (control === 'retry') await retryEventTask(eventId, task.id);
      if (control === 'resume') await resumeEventTask(eventId, task.id, response);
      setResumeTaskId(null);
      setResumeResponse('');
      setError('');
      await loadSurface();
      if (conversationId) {
        const conversation = await getConversation(conversationId);
        setMessages(conversation.messages);
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setTaskActionId(null);
    }
  }

  async function decide(proposalId: string, decision: 'approve' | 'reject') {
    await decideEventProposal(eventId, proposalId, decision);
    await loadSurface();
  }

  function openFile(file: EventFile) {
    openWorkbenchTab({ id: `file:${file.id}`, kind: 'file', label: file.name, file });
  }

  function openReplay(runId: string, taskTitle: string) {
    openWorkbenchTab({ id: `replay:${runId}`, kind: 'replay', label: `${taskTitle} replay`, runId });
  }

  function askAloyAboutFile(reference: StoredFileReference) {
    attachStoredFile(reference);
    setInput((current) => current || `Help me understand and work with ${reference.name}.`);
    setWorkspaceMode(window.matchMedia('(min-width: 768px)').matches ? 'split' : 'conversation');
  }

  if (!data) {
    return (
      <div className="flex h-full items-center justify-center">
        {error ? <p className="text-sm text-red-600">{error}</p> : <Spinner className="h-7 w-7" />}
      </div>
    );
  }

  const tasksSection = data.surface.sections.find((section) => section.kind === 'tasks');
  const filesSection = data.surface.sections.find((section) => section.kind === 'files');
  const tasks = tasksSection?.kind === 'tasks' ? tasksSection.tasks : [];
  const activity = trailEntries;
  const files = filesSection?.kind === 'files' ? filesSection.files : [];
  const openTasks = tasks.filter(
    (task) => task.status !== 'done' && task.status !== 'cancelled',
  ).length;
  const receipts = data.surface.execution_groups.flatMap((group) =>
    group.receipts.map((receipt, index) => ({ receipt, group, id: `${group.id}:${index}` })),
  );

  const tabs: Array<{ id: ContextTab; icon: typeof ListTodo; label: string; count?: number }> = [
    { id: 'tasks', icon: ListTodo, label: 'Tasks', count: openTasks },
    { id: 'approvals', icon: ShieldCheck, label: 'Approvals', count: data.surface.proposals.length },
    { id: 'receipts', icon: BadgeCheck, label: 'Receipts', count: receipts.length },
    { id: 'files', icon: FileText, label: 'Files', count: files.length },
    { id: 'trail', icon: Activity, label: 'Trail' },
  ];
  const lastMessage = messages.at(-1);

  return (
    <div className="relative flex h-full min-w-0 overflow-hidden bg-zinc-950">
      <section className="flex min-w-0 flex-1 flex-col">
        <header className="flex min-h-14 shrink-0 items-center gap-3 border-b border-zinc-800 px-4 lg:px-5">
          {!data.event.is_life && <EventCover event={data.event} className="h-9 w-12 shrink-0 rounded-lg border border-zinc-800" />}
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <h1 className="truncate font-display text-base font-semibold text-zinc-100">
                {data.event.title}
              </h1>
              <span className="shrink-0 rounded-full bg-accent-600/10 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-accent-700">
                {data.event.is_life ? 'Life' : data.event.phase || 'Active'}
              </span>
              <span
                className={`flex shrink-0 items-center gap-1 text-[10px] font-medium ${liveStatus === 'live' ? 'text-emerald-500' : liveStatus === 'offline' ? 'text-red-500' : 'text-amber-500'}`}
                title={`Event updates: ${liveStatus}`}
              >
                {liveStatus === 'live' ? <Wifi size={11} /> : <WifiOff size={11} />}
                {liveStatus}
              </span>
            </div>
            {data.event.summary && (
              <p className="truncate text-xs text-zinc-500">{data.event.summary}</p>
            )}
          </div>
          <div className="flex shrink-0 rounded-lg border border-zinc-800 bg-zinc-900 p-0.5" aria-label="Event workspace view">
            {(['conversation', 'split', 'workbench'] as const).map((mode) => (
              <button
                key={mode}
                type="button"
                onClick={() => setWorkspaceMode(mode)}
                className={`rounded-md px-2.5 py-1.5 text-[11px] font-medium capitalize transition-colors ${workspaceMode === mode ? 'bg-zinc-700 text-zinc-100 shadow-sm' : 'text-zinc-500 hover:text-zinc-300'} ${mode === 'split' ? 'hidden md:block' : ''}`}
                aria-pressed={workspaceMode === mode}
              >
                {mode}
              </button>
            ))}
          </div>
          <button
            type="button"
            onClick={() => setContextOpen((value) => !value)}
            className="rounded-lg p-2 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200"
            aria-label={contextOpen ? 'Close event context' : 'Open event context'}
            title={contextOpen ? 'Close event context' : 'Open event context'}
          >
            {contextOpen ? <PanelRightClose size={18} /> : <PanelRightOpen size={18} />}
          </button>
        </header>

        {error && (
          <div className="mx-4 mt-3 rounded-lg border border-red-500/25 bg-red-500/10 px-3 py-2 text-sm text-red-600">
            {error}
          </div>
        )}

        <div
          ref={workspaceRef}
          className={`flex min-h-0 flex-1 ${workspaceMode === 'split' ? 'flex-col md:flex-row' : ''}`}
        >
        {workspaceMode !== 'workbench' && (
          <div
            className={`flex min-h-0 min-w-0 flex-col ${workspaceMode === 'split' ? 'flex-none' : 'flex-1'}`}
            style={workspaceMode === 'split' ? { flexBasis: `${splitRatio}%` } : undefined}
          >
        <div className="min-h-0 flex-1 overflow-y-auto px-4 py-5 lg:px-6">
          <div className="mx-auto max-w-4xl">
            {loadingConversation ? (
              <div className="flex justify-center py-16"><Spinner className="h-7 w-7" /></div>
            ) : messages.length === 0 ? (
              <div className="flex min-h-[52vh] flex-col items-center justify-center text-center">
                <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-2xl border border-zinc-800 bg-zinc-900 text-accent-700">
                  <Send size={20} />
                </div>
                <h2 className="font-display text-xl font-semibold">Start working on {data.event.title}</h2>
                <p className="mt-2 max-w-md text-sm leading-6 text-zinc-500">
                  This conversation stays with the Event for its entire lifetime. Tasks, files, decisions, and evidence remain beside it.
                </p>
              </div>
            ) : (
              <MessageList
                messages={messages}
                streaming={streaming}
                streamText={streamText}
                streamStatus={streamStatus}
                streamActivity={streamActivity}
                streamPlan={streamPlan}
                streamTools={streamTools}
                streamStep={streamStep}
                clarify={clarify}
                onAnswerClarify={answerClarify}
                approval={approval}
                onDecideApproval={answerApproval}
                onOpenArtifact={openArtifact}
                onResend={sending ? undefined : resend}
                onContinue={sending ? undefined : continueRun}
                hasOlder={!!messageCursor}
                loadingOlder={loadingOlderMessages}
                onLoadOlder={() => void loadOlderMessages()}
                afterMessages={(
                  <SurfaceOpenCard
                    eventId={eventId}
                    eventTitle={data.event.title}
                    refreshKey={activity[0]?.id}
                    visible={
                      workspaceMode === 'conversation'
                      && !sending
                      && !streaming
                      && !clarify
                      && !approval
                      && lastMessage?.role === 'assistant'
                    }
                    onOpen={openSurface}
                  />
                )}
              />
            )}
          </div>
        </div>

        <div className="shrink-0 border-t border-zinc-800 bg-zinc-950/95 px-4 py-3 backdrop-blur lg:px-6">
          <div className="mx-auto max-w-4xl">
            <Composer
              value={input}
              onChange={setInput}
              onSend={handleSend}
              onAddFiles={addAttachments}
              pendingImages={pendingImages}
              onRemoveImage={removeImage}
              pendingFiles={pendingFiles}
              onRemoveFile={removeFile}
              disabled={sending && !clarify}
              placeholder={clarify ? 'Answer the question above…' : approval ? 'Approve or reject the action above…' : `Work on ${data.event.title}…`}
              attachFull={attachmentsFull}
              onStop={sending && !clarify ? stopRun : undefined}
            />
          </div>
        </div>
          </div>
        )}

        {workspaceMode === 'split' && (
          <button
            type="button"
            onPointerDown={startResize}
            className="group relative hidden w-1 shrink-0 cursor-col-resize bg-zinc-800 transition-colors hover:bg-accent-600 md:block"
            aria-label="Resize Conversation and Workbench"
            title="Drag to resize"
          >
            <span className="absolute left-1/2 top-1/2 h-10 w-0.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-zinc-600 group-hover:bg-white/70" />
          </button>
        )}

        {workspaceMode !== 'conversation' && conversationId && (
          <div className="min-h-0 min-w-0 flex-1 border-t border-zinc-800 md:border-t-0">
            <EventWorkbench
              eventId={eventId}
              eventTitle={data.event.title}
              conversationId={conversationId}
              refreshKey={activity[0]?.id}
              tabs={workbenchTabs}
              activeTabId={activeWorkbenchTabId}
              onSelectTab={setActiveWorkbenchTabId}
              onCloseTab={closeWorkbenchTab}
              onDismiss={() => setWorkspaceMode('conversation')}
              onAskAloy={askAloyAboutFile}
              showSurfaceAlongside={showSurfaceAlongside}
              onToggleSurfaceAlongside={() => setShowSurfaceAlongside((value) => !value)}
              resourceRatio={resourceRatio}
              onResourceRatioChange={setResourceRatio}
            />
          </div>
        )}
        </div>
      </section>

      {contextOpen && (
        <aside className="absolute inset-y-0 right-0 z-20 flex w-[min(420px,100%)] shrink-0 flex-col border-l border-zinc-800 bg-zinc-900 shadow-2xl xl:static xl:w-[390px] xl:shadow-none">
          <div className="flex h-14 shrink-0 items-center justify-between border-b border-zinc-800 px-4">
            <div>
              <p className="text-sm font-semibold text-zinc-100">Event context</p>
              <p className="text-[11px] text-zinc-500">Durable, trusted working state</p>
            </div>
            <button type="button" onClick={() => setContextOpen(false)} className="rounded-md p-1.5 text-zinc-500 hover:bg-zinc-800" aria-label="Close event context">
              <X size={17} />
            </button>
          </div>

          <div className="grid shrink-0 grid-cols-5 border-b border-zinc-800 px-2">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                type="button"
                onClick={() => setContextTab(tab.id)}
                className={`relative flex min-h-12 items-center justify-center gap-1.5 text-xs font-medium transition-colors ${contextTab === tab.id ? 'text-accent-700' : 'text-zinc-500 hover:text-zinc-300'}`}
              >
                <tab.icon size={15} />
                <span className="hidden 2xl:inline">{tab.label}</span>
                {!!tab.count && <span className="rounded-full bg-zinc-800 px-1.5 py-0.5 text-[10px]">{tab.count}</span>}
                {contextTab === tab.id && <span className="absolute inset-x-2 bottom-0 h-0.5 rounded-full bg-accent-600" />}
              </button>
            ))}
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto p-4">
            {contextTab === 'tasks' && (
              <div>
                <div className="flex gap-2">
                  <input
                    className={INPUT}
                    value={taskTitle}
                    onChange={(event) => setTaskTitle(event.target.value)}
                    onKeyDown={(event) => { if (event.key === 'Enter') void addTask(); }}
                    placeholder="Add a task"
                  />
                  <Button size="icon" onClick={addTask} disabled={!taskTitle.trim()} aria-label="Add task"><Plus size={16} /></Button>
                </div>
                <div className="mt-4 divide-y divide-zinc-800">
                  {tasks.map((task) => {
                    const active = task.status === 'queued' || task.status === 'in_progress';
                    const recoverable = task.status === 'failed' || task.status === 'cancelled';
                    const canResume = task.status === 'blocked' || task.status === 'waiting_approval';
                    return (
                      <div key={task.id} className="group py-3">
                        <div className="flex items-start gap-2.5">
                          <button type="button" onClick={() => void toggleTask(task.id, task.status)} disabled={!taskCanToggle(task.status)} className="mt-0.5 text-zinc-500 hover:text-accent-700 disabled:cursor-default disabled:hover:text-zinc-500" aria-label={task.status === 'open' ? 'Complete task' : task.status === 'done' ? 'Reopen task' : `Task is ${taskStatusLabel(task.status)}`}>
                            {task.status === 'done' ? <CheckCircle2 size={17} /> : <Circle size={17} />}
                          </button>
                          <div className="min-w-0 flex-1">
                            <p className={`text-sm leading-5 ${task.status === 'done' ? 'text-zinc-500 line-through' : 'text-zinc-300'}`}>{task.title}</p>
                            {task.status === 'queued' && <p className="mt-1 text-xs text-zinc-500">Waiting for this Event&apos;s work slot. You can leave Aloy open or closed.</p>}
                            {task.status === 'in_progress' && <p className="mt-1 text-xs text-accent-700">Aloy is working durably in the background.</p>}
                            {task.status === 'blocked' && <p className="mt-1 text-xs text-amber-700">Needs your input: {task.blocker || 'more information is required'}</p>}
                            {task.status === 'waiting_approval' && <p className="mt-1 text-xs text-amber-700">Waiting for a decision or committed receipt.</p>}
                            {task.status === 'failed' && <p className="mt-1 text-xs text-red-600">The Run failed safely. Retry starts a fresh Run.</p>}
                          </div>
                          {task.status !== 'open' && task.status !== 'done' && (
                            <span className="rounded-full bg-zinc-800 px-2 py-0.5 text-[10px] font-medium capitalize text-zinc-400">
                              {taskStatusLabel(task.status)}
                            </span>
                          )}
                          {!active && !canResume && (
                            <button type="button" onClick={() => void removeTask(task.id)} className="rounded p-1 text-zinc-600 opacity-0 hover:text-red-500 group-hover:opacity-100 focus:opacity-100" aria-label="Delete task"><Trash2 size={14} /></button>
                          )}
                        </div>
                        <div className="ml-7 mt-2 flex flex-wrap gap-2">
                          {task.status === 'open' && (
                            <Button size="sm" onClick={() => void runTaskControl(task, 'work')} disabled={taskActionId === task.id}><Play size={13} />Work on this</Button>
                          )}
                          {(active || canResume) && (
                            <Button size="sm" variant="ghost" onClick={() => void runTaskControl(task, 'stop')} disabled={taskActionId === task.id}><Square size={12} />Stop</Button>
                          )}
                          {recoverable && (
                            <Button size="sm" variant="outline" onClick={() => void runTaskControl(task, 'retry')} disabled={taskActionId === task.id}><RotateCcw size={13} />Retry</Button>
                          )}
                          {canResume && (
                            <Button size="sm" variant="outline" onClick={() => {
                              if (task.status === 'blocked') setResumeTaskId(task.id);
                              else void runTaskControl(task, 'resume');
                            }} disabled={taskActionId === task.id}><Play size={13} />Resume</Button>
                          )}
                        </div>
                        {resumeTaskId === task.id && (
                          <div className="ml-7 mt-2 flex gap-2">
                            <input className={INPUT} value={resumeResponse} onChange={(event) => setResumeResponse(event.target.value)} placeholder="Answer Aloy&apos;s question" autoFocus />
                            <Button size="sm" onClick={() => void runTaskControl(task, 'resume', resumeResponse)} disabled={!resumeResponse.trim() || taskActionId === task.id}>Continue</Button>
                          </div>
                        )}
                      </div>
                    );
                  })}
                  {tasks.length === 0 && <p className="py-8 text-center text-sm text-zinc-500">No tasks yet.</p>}
                </div>
              </div>
            )}

            {contextTab === 'approvals' && (
              <div className="space-y-3">
                {data.surface.proposals.map((proposal) => <ProposalCard key={proposal.id} proposal={proposal} onDecision={(decision) => decide(proposal.id, decision)} />)}
                {data.surface.proposals.length === 0 && <p className="py-8 text-center text-sm text-zinc-500">Nothing needs your decision.</p>}
              </div>
            )}

            {contextTab === 'receipts' && (
              <div className="space-y-3">
                {receipts.map(({ receipt, group, id }) => (
                  <article key={id} className="rounded-xl border border-emerald-500/20 bg-emerald-500/5 p-3">
                    <div className="flex items-start gap-2.5">
                      <BadgeCheck size={17} className="mt-0.5 shrink-0 text-emerald-500" />
                      <div className="min-w-0">
                        <p className="text-sm font-medium text-zinc-200">{String(receipt.action || receipt.type || receipt.status || 'Committed action')}</p>
                        <p className="mt-1 text-xs text-zinc-500">{group.task_title} · {when(group.created_at)}</p>
                        <pre className="mt-3 max-h-40 overflow-auto whitespace-pre-wrap rounded-lg bg-zinc-950/70 p-2 font-mono text-[10px] leading-4 text-zinc-500">{JSON.stringify(receipt, null, 2)}</pre>
                      </div>
                    </div>
                  </article>
                ))}
                {receipts.length === 0 && <p className="py-8 text-center text-sm text-zinc-500">No committed receipts yet.</p>}
              </div>
            )}

            {contextTab === 'files' && (
              <div className="space-y-2">
                {files.map((file) => (
                  <button type="button" key={file.id} onClick={() => openFile(file)} className="flex w-full items-center gap-3 rounded-lg border border-zinc-800 bg-zinc-950/50 p-3 text-left transition-colors hover:border-zinc-700 hover:bg-zinc-900">
                    <FileText size={17} className="shrink-0 text-zinc-500" />
                    <div className="min-w-0"><p className="truncate text-sm text-zinc-300">{file.name}</p><p className="text-xs text-zinc-500">{file.kind} · {Math.max(1, Math.round(file.size_bytes / 1024))} KB</p></div>
                  </button>
                ))}
                {files.length === 0 && <p className="py-8 text-center text-sm text-zinc-500">No Event files yet.</p>}
              </div>
            )}

            {contextTab === 'trail' && (
              <div className="space-y-4">
                {data.surface.execution_groups.map((group) => (
                  <details key={group.id} className="rounded-xl border border-zinc-800 bg-zinc-950/50 p-3">
                    <summary className="cursor-pointer list-none">
                      <div className="flex items-start justify-between gap-3">
                        <div>
                          <p className="text-sm font-medium text-zinc-200">{group.task_title}</p>
                          <p className="mt-1 text-xs text-zinc-500">Run {group.run_status} · {when(group.created_at)}</p>
                        </div>
                        <span className="rounded-full bg-zinc-800 px-2 py-0.5 text-[10px] text-zinc-400">{group.entries.length} updates</span>
                      </div>
                    </summary>
                    <div className="mt-3 space-y-3 border-t border-zinc-800 pt-3">
                      {group.entries.map((entry) => (
                        <div key={entry.id} className="border-l border-zinc-700 pl-3">
                          <p className="text-xs text-zinc-300">{entry.summary}</p>
                          <p className="mt-1 text-[11px] text-zinc-600">{when(entry.created_at)}</p>
                        </div>
                      ))}
                      <div className="flex flex-wrap gap-3 text-xs">
                        <button type="button" onClick={() => openReplay(group.run_id, group.task_title)} className="text-accent-700 hover:text-accent-600">Open Run replay</button>
                        {group.conversation_id && <Link to={`/chat/${group.conversation_id}`} className="text-zinc-400 hover:text-zinc-200">Origin conversation</Link>}
                        {group.artifacts.map((artifact) => <button type="button" key={artifact.id} onClick={() => openFile(artifact)} className="text-zinc-400 hover:text-zinc-200">{artifact.name}</button>)}
                        {!!group.proposals.length && <span className="text-zinc-500">{group.proposals.length} proposal{group.proposals.length === 1 ? '' : 's'}</span>}
                        {!!group.receipts.length && <span className="text-emerald-600">{group.receipts.length} receipt{group.receipts.length === 1 ? '' : 's'}</span>}
                      </div>
                    </div>
                  </details>
                ))}
                {data.surface.execution_groups.length > 0 && activity.length > 0 && (
                  <p className="pt-2 text-[11px] font-semibold uppercase tracking-wider text-zinc-600">All Event activity</p>
                )}
                {activity.map((entry) => (
                  <div key={entry.id} className="relative border-l border-zinc-700 pl-4">
                    <span className="absolute -left-1 top-1 h-2 w-2 rounded-full bg-zinc-600" />
                    <p className="text-sm leading-5 text-zinc-300">{entry.summary}</p>
                    <p className="mt-1 text-xs text-zinc-500">{when(entry.created_at)}</p>
                  </div>
                ))}
                {activity.length === 0 && <p className="py-8 text-center text-sm text-zinc-500">No activity yet.</p>}
                {trailCursor && (
                  <Button variant="ghost" className="w-full" onClick={() => void loadOlderTrail()} disabled={loadingOlderTrail}>
                    {loadingOlderTrail ? 'Loading…' : 'Load older activity'}
                  </Button>
                )}
              </div>
            )}
          </div>
        </aside>
      )}

      {!contextOpen && (
        <aside className="hidden w-12 shrink-0 flex-col items-center border-l border-zinc-800 bg-zinc-900 py-2 xl:flex" aria-label="Collapsed Event context">
          <button type="button" onClick={() => setContextOpen(true)} className="mb-2 rounded-lg p-2 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200" title="Open Event context" aria-label="Open Event context"><PanelRightOpen size={17} /></button>
          <div className="h-px w-7 bg-zinc-800" />
          {tabs.map((tab) => (
            <button key={tab.id} type="button" onClick={() => { setContextTab(tab.id); setContextOpen(true); }} className="relative mt-2 rounded-lg p-2 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200" title={tab.label} aria-label={`Open ${tab.label}`}>
              <tab.icon size={16} />
              {!!tab.count && <span className="absolute -right-1 -top-1 min-w-4 rounded-full bg-accent-700 px-1 text-center text-[9px] font-semibold text-white">{tab.count}</span>}
            </button>
          ))}
        </aside>
      )}
    </div>
  );
}
