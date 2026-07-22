import {
  useCallback,
  useRef,
  useState,
  type Dispatch,
  type SetStateAction,
} from 'react';
import { stopGeneration } from '@/api/conversations';
import {
  attachLiveRun,
  streamMessage,
  submitApproval,
  submitClarification,
  type ApprovalDecision,
  type SSECallbacks,
} from '@/api/sse';
import type {
  MessageImage,
  MessageResponse,
  PendingFile,
  PlanItem,
  RunTimelineEvent,
  RunTimelineKind,
  SSEMessageEvent,
} from '@/types';

export interface ClarifyState {
  id: string;
  question: string;
  options: string[];
}

export interface ApprovalState {
  id: string;
  tool: string;
  arguments: Record<string, unknown>;
  description: string;
  allowedDecisions: string[];
}

export interface UseStreamingRunParams {
  /** Conversation the next send/stop/reattach targets (null = none active). */
  activeId: string | null;
  /** The caller's message-list setter — the stream appends into it. */
  setMessages: Dispatch<SetStateAction<MessageResponse[]>>;
  /** Called when a run finishes so the caller can refresh its listings. */
  onConversationsRefresh: () => void;
}

function liveStoryEvent(
  kind: RunTimelineKind,
  publicPayload: Record<string, unknown>,
  sequence: number,
): RunTimelineEvent {
  return {
    id: `live-${sequence}-${Date.now()}`,
    sequence,
    kind,
    schema_version: 1,
    public_payload: publicPayload,
    created_at: new Date().toISOString(),
  };
}

/**
 * Owns one streaming run: the SSE lifecycle (send → frames → final message),
 * its full UI state slice, stop/resend/continue/clarify controls, and the
 * re-attach flow for runs still in flight after navigating away and back.
 *
 * Page-agnostic by design (Surfaces will reuse it): it never touches
 * routing — the caller wires route changes to `abortStream`/`resetStreamUi`
 * and passes ids in.
 */
export function useStreamingRun({
  activeId,
  setMessages,
  onConversationsRefresh,
}: UseStreamingRunParams) {
  const [sending, setSending] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [streamStory, setStreamStory] = useState<RunTimelineEvent[]>([]);
  const [streamText, setStreamText] = useState('');
  const [clarify, setClarify] = useState<ClarifyState | null>(null);
  const [approval, setApproval] = useState<ApprovalState | null>(null);
  // The in-flight stream's abort handle — aborted on conversation switch and
  // on unmount so a stream can never bleed into another conversation's view.
  const streamAbortRef = useRef<AbortController | null>(null);
  const streamStoryRef = useRef<RunTimelineEvent[]>([]);

  function replaceStreamStory(
    update: (previous: RunTimelineEvent[]) => RunTimelineEvent[],
  ) {
    setStreamStory((previous) => {
      const next = update(previous);
      streamStoryRef.current = next;
      return next;
    });
  }

  function appendStory(
    kind: RunTimelineKind,
    publicPayload: Record<string, unknown>,
  ) {
    replaceStreamStory((previous) => [
      ...previous,
      liveStoryEvent(kind, publicPayload, previous.length + 1),
    ]);
  }

  const resetStreamUi = useCallback(() => {
    setStreaming(false);
    setSending(false);
    setStreamStory([]);
    streamStoryRef.current = [];
    setStreamText('');
    setClarify(null);
    setApproval(null);
  }, []);

  /** Kill the in-flight stream (conversation switch / page unmount). */
  const abortStream = useCallback(() => {
    streamAbortRef.current?.abort();
    streamAbortRef.current = null;
  }, []);

  /** Shared stream callbacks — used by a fresh send AND by re-attaching to an
   *  in-flight run after navigating back. */
  function buildStreamCallbacks(): SSECallbacks {
    return {
      onText: (text) => setStreamText((prev) => prev + text),
      onRunStart: () => {
        replaceStreamStory((previous) =>
          previous.some((entry) => entry.kind === 'run_started')
            ? previous
            : [liveStoryEvent('run_started', { status: 'running' }, 1)],
        );
      },
      onRunEnd: (payload) => appendStory('run_finished', payload),
      onActivity: (activity) => {
        if (activity.trim()) appendStory('activity_changed', { activity });
      },
      onPlan: (plan, summary) =>
        appendStory('plan_changed', {
          plan: plan as PlanItem[],
          summary,
        }),
      onToolStart: (payload) => {
        const name = String(payload.name ?? 'tool');
        appendStory('action_started', {
          call_id: String(payload.call_id ?? ''),
          label: String(payload.label ?? `Running ${name}`),
        });
      },
      onToolEnd: (payload) => {
        const storyPayload: Record<string, unknown> = {
          call_id: String(payload.call_id ?? ''),
          label: String(
            payload.label ?? `Finished ${String(payload.name ?? 'action')}`,
          ),
          success: Boolean(payload.success),
        };
        if (typeof payload.duration_seconds === 'number') {
          storyPayload.duration_seconds = payload.duration_seconds;
        }
        appendStory('action_finished', storyPayload);
      },
      onClarification: (req) => {
        appendStory('attention_required', {
          request_id: req.id,
          request_kind: 'clarification',
          description: req.question,
        });
        setClarify({ id: req.id, question: req.question, options: req.options });
      },
      onApproval: (req) => {
        appendStory('attention_required', {
          request_id: req.id,
          request_kind: 'approval',
          description: req.description,
        });
        setApproval({
          id: req.id,
          tool: req.tool,
          arguments: req.arguments,
          description: req.description,
          allowedDecisions: req.allowed_decisions,
        });
      },
      onMessage: (data: SSEMessageEvent) => {
        const assistantMsg: MessageResponse = {
          id: `msg-${Date.now()}`,
          role: 'assistant',
          content: data.content,
          metadata: {
            run_id: data.run_id || null,
            reasoning: data.reasoning || null,
            steps_taken: data.steps_taken,
            metrics: data.metrics || null,
            artifacts: data.artifacts || [],
            plan: data.plan || [],
            selected_skills: data.selected_skills || [],
            work_story: streamStoryRef.current,
            ...(data.stopped ? { stopped: true } : {}),
          },
          created_at: new Date().toISOString(),
        };
        // Dedupe by run_id: on re-attach the persisted message may already be
        // in the loaded history when the replayed 'message' frame arrives.
        setMessages((prev) =>
          data.run_id && prev.some((m) => m.metadata?.run_id === data.run_id)
            ? prev
            : [...prev, assistantMsg],
        );
        // Move the Work Story onto the final message before live state clears.
        setStreamText('');
        setStreaming(false);
        setStreamStory([]);
        streamStoryRef.current = [];
      },
      onError: (err) => {
        const failedStory = [
          ...streamStoryRef.current,
          liveStoryEvent(
            'run_failed',
            { message: 'The run stopped unexpectedly.' },
            streamStoryRef.current.length + 1,
          ),
        ];
        const errMsg: MessageResponse = {
          id: `err-${Date.now()}`,
          role: 'assistant',
          content: `Error: ${err}`,
          metadata: { work_story: failedStory },
          created_at: new Date().toISOString(),
        };
        setMessages((prev) => [...prev, errMsg]);
        // Fully reset — an error frame may arrive with the stream held open,
        // and relying on a later onDone left the UI locked in 'sending'.
        resetStreamUi();
      },
      onDone: () => {
        resetStreamUi();
        onConversationsRefresh();
      },
    };
  }

  /** If this conversation has an in-flight run (we navigated away and came
   *  back), re-attach: the server replays all frames, then continues live. */
  async function tryReattach(convId: string) {
    const controller = new AbortController();
    try {
      const attached = await attachLiveRun(
        convId,
        buildStreamCallbacks(),
        controller.signal,
        () => {
          streamAbortRef.current = controller;
          setSending(true);
          setStreaming(true);
          const initial = [
            liveStoryEvent('run_started', { status: 'resuming' }, 1),
          ];
          streamStoryRef.current = initial;
          setStreamStory(initial);
        },
      );
      if (!attached) return;
    } catch {
      resetStreamUi();
    } finally {
      if (streamAbortRef.current === controller) {
        streamAbortRef.current = null;
      }
    }
  }

  async function dispatchSend(
    content: string,
    images: MessageImage[],
    files: PendingFile[],
    opts?: {
      resumeRunId?: string;
      surfaceSelection?: Record<string, unknown>;
    },
  ) {
    if (!activeId) return;
    setSending(true);
    setStreaming(true);
    const initialStory = [
      liveStoryEvent('run_started', { status: 'running' }, 1),
    ];
    streamStoryRef.current = initialStory;
    setStreamStory(initialStory);
    setStreamText('');
    setClarify(null);
    setApproval(null);

    const userMsg: MessageResponse = {
      id: `temp-${Date.now()}`,
      role: 'user',
      content,
      metadata:
        images.length > 0 || files.length > 0 || opts?.surfaceSelection
          ? {
              ...(images.length > 0 ? { images } : {}),
              ...(files.length > 0
                ? {
                    files: files.map((f) => ({
                      name: f.name,
                      size: f.size,
                      // Keep the durable ref so the chip's download/bookmark
                      // icons work on the OPTIMISTIC message too, not only
                      // after a reload.
                      ...(f.file_id ? { file_id: f.file_id } : {}),
                    })),
                  }
                : {}),
              ...(opts?.surfaceSelection
                ? { surface_selection: opts.surfaceSelection }
                : {}),
            }
          : null,
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, userMsg]);

    const callbacks = buildStreamCallbacks();

    // Abortable stream: switching conversations or leaving the page aborts it
    // so tokens can't bleed into another conversation's view.
    const controller = new AbortController();
    streamAbortRef.current = controller;
    try {
      const textFiles = files.filter((f) => f.content != null);
      const binaryDocs = files.filter((f) => f.data != null);
      const uploadRefs = files.filter((f) => f.file_id).map((f) => f.file_id!);
      await streamMessage(activeId, content, callbacks, {
        resume_run_id: opts?.resumeRunId,
        file_refs: uploadRefs.length > 0 ? uploadRefs : undefined,
        surface_selection: opts?.surfaceSelection,
        images: images.length > 0 ? images : undefined,
        files:
          textFiles.length > 0
            ? textFiles.map((f) => ({ name: f.name, content: f.content! }))
            : undefined,
        documents:
          binaryDocs.length > 0
            ? binaryDocs.map((f) => ({
                name: f.name,
                data: f.data!,
                media_type: f.media_type!,
              }))
            : undefined,
        signal: controller.signal,
      });
    } catch {
      resetStreamUi();
    } finally {
      if (streamAbortRef.current === controller) {
        streamAbortRef.current = null;
      }
    }
  }

  /** Re-send an earlier user message as a fresh turn (from its bubble). */
  async function resend(content: string) {
    if (!activeId || sending || clarify || !content.trim()) return;
    await dispatchSend(content.trim(), [], []);
  }

  /** Continue an interrupted response. If the stopped run's state is still
   *  warm on the server this truly resumes it (tool work intact); otherwise
   *  the continuation instruction runs as a normal turn over history. */
  async function continueRun(message: MessageResponse) {
    if (!activeId || sending || clarify) return;
    const runId = message.metadata?.run_id ?? undefined;
    await dispatchSend(
      'Continue your interrupted response from exactly where it stopped. Do not repeat what you already wrote.',
      [],
      [],
      runId ? { resumeRunId: runId } : undefined,
    );
  }

  /** Stop the in-flight run: the agent halts at the next step boundary and
   *  the stream finishes with a final frame (which resets the UI). */
  async function stopRun() {
    if (!activeId) return;
    appendStory('activity_changed', { activity: 'Stopping safely' });
    try {
      await stopGeneration(activeId);
    } catch {
      // 404 = the run already finished; the stream's done frame cleans up.
    }
  }

  async function answerClarify(value: string) {
    if (!clarify) return;
    const id = clarify.id;
    setClarify(null);
    try {
      await submitClarification(id, value);
    } catch {
      // the error surfaces via the stream
    }
  }

  /** Approve or reject the paused consequential tool. The run resumes (runs the
   *  tool) or skips it and continues, then streams on. */
  async function answerApproval(decision: ApprovalDecision) {
    if (!approval) return;
    const id = approval.id;
    setApproval(null);
    appendStory('activity_changed', {
      activity: decision.type === 'approve' ? 'Continuing approved work' : 'Skipping that action',
    });
    try {
      await submitApproval(id, decision);
    } catch {
      // the error surfaces via the stream
    }
  }

  return {
    sending,
    streaming,
    streamStory,
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
  };
}
