import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Activity,
  CheckCircle2,
  Circle,
  FileText,
  ListTodo,
  PanelRightClose,
  PanelRightOpen,
  Plus,
  Send,
  ShieldCheck,
  Trash2,
  X,
} from 'lucide-react';
import { useParams } from 'react-router-dom';
import { getConversation } from '@/api/conversations';
import {
  createEventTask,
  decideEventProposal,
  deleteEventTask,
  getEventSurface,
  updateEventTask,
  type EventSurfaceResponse,
  type EventTask,
} from '@/api/events';
import { ArtifactDrawer } from '@/components/chat/ArtifactDrawer';
import { Composer } from '@/components/chat/Composer';
import { MessageList } from '@/components/chat/MessageList';
import { ProposalCard } from '@/components/events/ProposalCard';
import { Button } from '@/components/ui/Button';
import { Spinner } from '@/components/ui/Spinner';
import { useAttachments } from '@/hooks/useAttachments';
import { useStreamingRun } from '@/hooks/useStreamingRun';
import type { MessageResponse } from '@/types';

type ContextTab = 'tasks' | 'decisions' | 'files' | 'trail';

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
  const [data, setData] = useState<EventSurfaceResponse | null>(null);
  const [messages, setMessages] = useState<MessageResponse[]>([]);
  const [input, setInput] = useState('');
  const [taskTitle, setTaskTitle] = useState('');
  const [artifactPath, setArtifactPath] = useState<string | null>(null);
  const [loadingConversation, setLoadingConversation] = useState(true);
  const [contextOpen, setContextOpen] = useState(true);
  const [contextTab, setContextTab] = useState<ContextTab>('tasks');
  const [error, setError] = useState('');
  const previousSending = useRef(false);

  const conversationId = data?.event.conversation_id ?? null;

  const loadSurface = useCallback(async () => {
    if (!eventId) return;
    try {
      setData(await getEventSurface(eventId));
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
    setArtifactPath(null);
    setLoadingConversation(true);
    let cancelled = false;
    getConversation(conversationId)
      .then((conversation) => {
        if (cancelled) return;
        setMessages(conversation.messages);
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

  async function decide(proposalId: string, decision: 'approve' | 'reject') {
    await decideEventProposal(eventId, proposalId, decision);
    await loadSurface();
  }

  if (!data) {
    return (
      <div className="flex h-full items-center justify-center">
        {error ? <p className="text-sm text-red-600">{error}</p> : <Spinner className="h-7 w-7" />}
      </div>
    );
  }

  const tasksSection = data.surface.sections.find((section) => section.kind === 'tasks');
  const activitySection = data.surface.sections.find((section) => section.kind === 'activity');
  const filesSection = data.surface.sections.find((section) => section.kind === 'files');
  const tasks = tasksSection?.kind === 'tasks' ? tasksSection.tasks : [];
  const activity = activitySection?.kind === 'activity' ? activitySection.entries : [];
  const files = filesSection?.kind === 'files' ? filesSection.files : [];
  const openTasks = tasks.filter(
    (task) => task.status !== 'done' && task.status !== 'cancelled',
  ).length;

  const tabs: Array<{ id: ContextTab; icon: typeof ListTodo; label: string; count?: number }> = [
    { id: 'tasks', icon: ListTodo, label: 'Tasks', count: openTasks },
    { id: 'decisions', icon: ShieldCheck, label: 'Decisions', count: data.surface.proposals.length },
    { id: 'files', icon: FileText, label: 'Files', count: files.length },
    { id: 'trail', icon: Activity, label: 'Trail' },
  ];

  return (
    <div className="relative flex h-full min-w-0 overflow-hidden bg-zinc-950">
      <section className="flex min-w-0 flex-1 flex-col">
        <header className="flex min-h-14 shrink-0 items-center gap-3 border-b border-zinc-800 px-4 lg:px-5">
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <h1 className="truncate font-display text-base font-semibold text-zinc-100">
                {data.event.title}
              </h1>
              <span className="shrink-0 rounded-full bg-accent-600/10 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-accent-700">
                {data.event.is_life ? 'Life' : data.event.phase || 'Active'}
              </span>
            </div>
            {data.event.summary && (
              <p className="truncate text-xs text-zinc-500">{data.event.summary}</p>
            )}
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
                onOpenArtifact={setArtifactPath}
                onResend={sending ? undefined : resend}
                onContinue={sending ? undefined : continueRun}
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

          <div className="grid shrink-0 grid-cols-4 border-b border-zinc-800 px-2">
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
                  {tasks.map((task) => (
                    <div key={task.id} className="group flex items-start gap-2.5 py-3">
                      <button type="button" onClick={() => void toggleTask(task.id, task.status)} disabled={!taskCanToggle(task.status)} className="mt-0.5 text-zinc-500 hover:text-accent-700 disabled:cursor-default disabled:hover:text-zinc-500" aria-label={task.status === 'open' ? 'Complete task' : task.status === 'done' ? 'Reopen task' : `Task is ${taskStatusLabel(task.status)}`}>
                        {task.status === 'done' ? <CheckCircle2 size={17} /> : <Circle size={17} />}
                      </button>
                      <span className={`min-w-0 flex-1 text-sm leading-5 ${task.status === 'done' ? 'text-zinc-500 line-through' : 'text-zinc-300'}`}>{task.title}</span>
                      {task.status !== 'open' && task.status !== 'done' && (
                        <span className="rounded-full bg-zinc-800 px-2 py-0.5 text-[10px] font-medium capitalize text-zinc-400">
                          {taskStatusLabel(task.status)}
                        </span>
                      )}
                      <button type="button" onClick={() => void removeTask(task.id)} className="rounded p-1 text-zinc-600 opacity-0 hover:text-red-500 group-hover:opacity-100 focus:opacity-100" aria-label="Delete task"><Trash2 size={14} /></button>
                    </div>
                  ))}
                  {tasks.length === 0 && <p className="py-8 text-center text-sm text-zinc-500">No tasks yet.</p>}
                </div>
              </div>
            )}

            {contextTab === 'decisions' && (
              <div className="space-y-3">
                {data.surface.proposals.map((proposal) => <ProposalCard key={proposal.id} proposal={proposal} onDecision={(decision) => decide(proposal.id, decision)} />)}
                {data.surface.proposals.length === 0 && <p className="py-8 text-center text-sm text-zinc-500">Nothing needs your decision.</p>}
              </div>
            )}

            {contextTab === 'files' && (
              <div className="space-y-2">
                {files.map((file) => (
                  <div key={file.id} className="flex items-center gap-3 rounded-lg border border-zinc-800 bg-zinc-950/50 p-3">
                    <FileText size={17} className="shrink-0 text-zinc-500" />
                    <div className="min-w-0"><p className="truncate text-sm text-zinc-300">{file.name}</p><p className="text-xs text-zinc-500">{file.kind} · {Math.max(1, Math.round(file.size_bytes / 1024))} KB</p></div>
                  </div>
                ))}
                {files.length === 0 && <p className="py-8 text-center text-sm text-zinc-500">No Event files yet.</p>}
              </div>
            )}

            {contextTab === 'trail' && (
              <div className="space-y-4">
                {activity.map((entry) => (
                  <div key={entry.id} className="relative border-l border-zinc-700 pl-4">
                    <span className="absolute -left-1 top-1 h-2 w-2 rounded-full bg-zinc-600" />
                    <p className="text-sm leading-5 text-zinc-300">{entry.summary}</p>
                    <p className="mt-1 text-xs text-zinc-500">{when(entry.created_at)}</p>
                  </div>
                ))}
                {activity.length === 0 && <p className="py-8 text-center text-sm text-zinc-500">No activity yet.</p>}
              </div>
            )}
          </div>
        </aside>
      )}

      {artifactPath && conversationId && <ArtifactDrawer conversationId={conversationId} openPath={artifactPath} onClose={() => setArtifactPath(null)} />}
    </div>
  );
}
