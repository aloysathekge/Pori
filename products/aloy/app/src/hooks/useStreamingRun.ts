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
  submitClarification,
  type SSECallbacks,
} from '@/api/sse';
import type {
  MessageImage,
  MessageResponse,
  PendingFile,
  PlanItem,
  SSEMessageEvent,
  SSEToolEvent,
} from '@/types';

export interface ClarifyState {
  id: string;
  question: string;
  options: string[];
}

export interface UseStreamingRunParams {
  /** Conversation the next send/stop/reattach targets (null = none active). */
  activeId: string | null;
  /** The caller's message-list setter — the stream appends into it. */
  setMessages: Dispatch<SetStateAction<MessageResponse[]>>;
  /** Called when a run finishes so the caller can refresh its listings. */
  onConversationsRefresh: () => void;
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
  const [streamStatus, setStreamStatus] = useState('');
  const [streamActivity, setStreamActivity] = useState('');
  const [streamPlan, setStreamPlan] = useState<PlanItem[]>([]);
  const [streamTools, setStreamTools] = useState<SSEToolEvent[]>([]);
  const [streamStep, setStreamStep] = useState<
    { step: number; max_steps: number } | undefined
  >(undefined);
  const [streamText, setStreamText] = useState('');
  const [clarify, setClarify] = useState<ClarifyState | null>(null);
  // The in-flight stream's abort handle — aborted on conversation switch and
  // on unmount so a stream can never bleed into another conversation's view.
  const streamAbortRef = useRef<AbortController | null>(null);

  const resetStreamUi = useCallback(() => {
    setStreaming(false);
    setSending(false);
    setStreamStatus('');
    setStreamActivity('');
    setStreamPlan([]);
    setStreamTools([]);
    setStreamStep(undefined);
    setStreamText('');
    setClarify(null);
  }, []);

  /** Kill the in-flight stream (conversation switch / page unmount). */
  const abortStream = useCallback(() => {
    streamAbortRef.current?.abort();
    streamAbortRef.current = null;
  }, []);

  /** Shared stream callbacks — used by a fresh send AND by re-attaching to an
   *  in-flight run after navigating back. */
  function buildStreamCallbacks(): SSECallbacks {
    let firstText = true;
    return {
      onText: (text) => {
        if (firstText) {
          firstText = false;
          setStreamStatus('Writing…');
          setStreamActivity('');
        }
        setStreamText((prev) => prev + text);
      },
      onStep: (info) => {
        setStreamStep(info);
        setStreamStatus('Thinking…');
      },
      onToolStart: (payload) => {
        const name = String(payload.name ?? 'tool');
        setStreamActivity(`Running ${name}…`);
        setStreamTools((prev) => [
          ...prev,
          { step: 0, tool: name, preview: '', success: false },
        ]);
      },
      onToolEnd: (payload) =>
        setStreamTools((prev) => {
          const next = [...prev];
          const name = String(payload.name ?? '');
          for (let i = next.length - 1; i >= 0; i--) {
            const entry = next[i];
            if (entry && entry.tool === name) {
              next[i] = { ...entry, success: Boolean(payload.success) };
              break;
            }
          }
          return next;
        }),
      onClarification: (req) =>
        setClarify({ id: req.id, question: req.question, options: req.options }),
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
        // Tear down the whole streaming UI (bubble + indicator) the instant the
        // final message lands, so nothing lingers/overlaps beneath it.
        setStreamText('');
        setStreaming(false);
        setStreamStatus('');
        setStreamActivity('');
        setStreamPlan([]);
        setStreamTools([]);
        setStreamStep(undefined);
      },
      onError: (err) => {
        const errMsg: MessageResponse = {
          id: `err-${Date.now()}`,
          role: 'assistant',
          content: `Error: ${err}`,
          metadata: null,
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
          setStreamStatus('Resuming…');
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
    opts?: { resumeRunId?: string },
  ) {
    if (!activeId) return;
    setSending(true);
    setStreaming(true);
    setStreamStatus('Thinking…');
    setStreamActivity('');
    setStreamPlan([]);
    setStreamTools([]);
    setStreamStep(undefined);
    setStreamText('');
    setClarify(null);

    const userMsg: MessageResponse = {
      id: `temp-${Date.now()}`,
      role: 'user',
      content,
      metadata:
        images.length > 0 || files.length > 0
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
    setStreamStatus('Stopping…');
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

  return {
    sending,
    streaming,
    streamStatus,
    streamActivity,
    streamPlan,
    streamTools,
    streamStep,
    streamText,
    clarify,
    resetStreamUi,
    abortStream,
    dispatchSend,
    tryReattach,
    resend,
    continueRun,
    stopRun,
    answerClarify,
  };
}
