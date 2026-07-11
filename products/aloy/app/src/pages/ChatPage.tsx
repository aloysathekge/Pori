import { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Plus, Send } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Spinner } from '@/components/ui/Spinner';
import { ConversationSwitcher } from '@/components/chat/ConversationSwitcher';
import { Composer } from '@/components/chat/Composer';
import { MessageBubble } from '@/components/chat/MessageBubble';
import { ArtifactDrawer } from '@/components/chat/ArtifactDrawer';
import { StreamingIndicator } from '@/components/chat/StreamingIndicator';
import {
  listConversations,
  getConversation,
  createConversation,
  deleteConversation,
  stopGeneration,
  uploadConversationFile,
} from '@/api/conversations';
import {
  attachLiveRun,
  streamMessage,
  submitClarification,
  type SSECallbacks,
} from '@/api/sse';
import type {
  ConversationResponse,
  MessageFile,
  MessageImage,
  MessageResponse,
  SSEMessageEvent,
  SSEToolEvent,
  PlanItem,
} from '@/types';

const MAX_IMAGES = 3;
const MAX_IMAGE_BYTES = 5 * 1024 * 1024; // 5MB per image (backend-enforced too)
const MAX_FILES = 3;
const MAX_FILE_CHARS = 200_000; // ~200KB of text (backend-enforced too)
const DOC_MIMES: Record<string, string> = {
  pdf: 'application/pdf',
  docx: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  xlsx: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
};
const MAX_DOC_BYTES = 10 * 1024 * 1024; // 10MB per document
const MAX_UPLOAD_BYTES = 100 * 1024 * 1024; // durable-upload rung cap (backend too)
const TEXT_EXTENSIONS =
  /\.(txt|md|markdown|csv|tsv|json|jsonl|js|jsx|ts|tsx|py|rb|go|rs|java|c|h|cpp|cs|php|html|css|scss|xml|yml|yaml|toml|ini|cfg|conf|env|sh|bash|ps1|sql|log|diff|patch)$/i;

export function ChatPage() {
  const { conversationId: routeConversationId } = useParams();
  const navigate = useNavigate();
  const [conversations, setConversations] = useState<ConversationResponse[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<MessageResponse[]>([]);
  const [artifactPath, setArtifactPath] = useState<string | null>(null);
  const [input, setInput] = useState('');
  const [pendingImages, setPendingImages] = useState<MessageImage[]>([]);
  const [pendingFiles, setPendingFiles] = useState<
    (MessageFile & {
      content?: string;
      data?: string;
      media_type?: string;
      // Durable-upload rung: set while/after uploading to object storage.
      key?: string;
      file_id?: string;
      uploading?: boolean;
      progress?: number;
      error?: boolean;
    })[]
  >([]);
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
  const [clarify, setClarify] = useState<{
    id: string;
    question: string;
    options: string[];
  } | null>(null);
  const [loadingConversation, setLoadingConversation] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
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

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, streaming, scrollToBottom]);

  useEffect(() => {
    loadConversations();
  }, []);

  useEffect(() => {
    // Kill any in-flight stream from the previous conversation and reset the
    // streaming UI + artifact drawer before rendering the new one.
    streamAbortRef.current?.abort();
    streamAbortRef.current = null;
    // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional reset on route change
    resetStreamUi();
    setArtifactPath(null);

    let cancelled = false;
    if (routeConversationId) {
      setActiveId(routeConversationId);
      setLoadingConversation(true);
      getConversation(routeConversationId)
        .then((detail) => {
          if (cancelled) return;
          setMessages(detail.messages);
          // Resume live streaming if this conversation has an in-flight run.
          void tryReattach(routeConversationId);
        })
        .catch(() => {
          if (!cancelled) setMessages([]);
        })
        .finally(() => {
          if (!cancelled) setLoadingConversation(false);
        });
    } else {
      setActiveId(null);
      setMessages([]);
    }
    return () => {
      cancelled = true;
    };
    // tryReattach is deliberately NOT a dependency: it closes over per-render
    // callbacks, and this effect must re-run only on route change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [routeConversationId, resetStreamUi]);

  // Abort the stream when leaving the chat page entirely.
  useEffect(() => {
    return () => streamAbortRef.current?.abort();
  }, []);

  // Landing on /chat with no conversation selected: open the most recent one
  // (nav 'Chat' should drop you into your chat, not an empty state).
  useEffect(() => {
    const mostRecent = conversations[0];
    if (!routeConversationId && mostRecent) {
      navigate(`/chat/${mostRecent.id}`, { replace: true });
    }
  }, [routeConversationId, conversations, navigate]);

  async function loadConversations() {
    try {
      const convos = await listConversations(50);
      setConversations(convos);
    } catch {
      // handle silently
    }
  }

  /** Durable-upload rung: too big (or too binary) for inline/native — the
   *  file streams to object storage NOW (eager, while the user types) and
   *  the message will carry only its reference. */
  /** Durable copy for an existing chip: streams the bytes to object storage
   *  and attaches file_id to the chip (which is what makes it bookmarkable
   *  into My Files). Failure just clears the uploading flag — the chip still
   *  works through its inline/native ride. */
  function uploadDurable(file: File, key: string) {
    const convId = activeId;
    if (!convId) return;
    uploadConversationFile(convId, file, (pct) =>
      setPendingFiles((prev) =>
        prev.map((f) => (f.key === key ? { ...f, progress: pct } : f)),
      ),
    )
      .then((res) =>
        setPendingFiles((prev) =>
          prev.map((f) =>
            f.key === key
              ? { ...f, uploading: false, progress: 100, file_id: res.file_id }
              : f,
          ),
        ),
      )
      .catch((err) => {
        console.error('[aloy] durable upload failed:', err);
        setPendingFiles((prev) =>
          prev.map((f) =>
            f.key === key ? { ...f, uploading: false, error: true } : f,
          ),
        );
      });
  }

  function uploadAttachment(file: File) {
    if (!activeId || file.size > MAX_UPLOAD_BYTES) return;
    if (pendingFiles.length >= MAX_FILES) return;
    const key = `${file.name}-${Date.now()}-${Math.random()}`;
    setPendingFiles((prev) =>
      prev.length >= MAX_FILES
        ? prev
        : [
            ...prev,
            { key, name: file.name, size: file.size, uploading: true, progress: 0 },
          ],
    );
    uploadDurable(file, key);
  }

  /** Route attachments: images render for the model's eyes; text-like files
   *  embed their content into the task; everything bigger or binary takes
   *  the durable-upload rung. Docs and text files ALSO get a durable copy
   *  (same chip), so any attachment can be saved to My Files — and later
   *  turns can still reach the bytes in the sandbox. */
  function addAttachments(files: Iterable<File>) {
    for (const file of files) {
      if (file.type.startsWith('image/')) {
        if (file.size > MAX_IMAGE_BYTES) {
          uploadAttachment(file);
          continue;
        }
        const reader = new FileReader();
        reader.onload = () => {
          const base64 = String(reader.result || '').split(',')[1] ?? '';
          if (!base64) return;
          setPendingImages((prev) =>
            prev.length >= MAX_IMAGES
              ? prev
              : [...prev, { data: base64, media_type: file.type }],
          );
        };
        reader.readAsDataURL(file);
      } else if (DOC_MIMES[file.name.split('.').pop()?.toLowerCase() ?? '']) {
        if (file.size > MAX_DOC_BYTES) {
          uploadAttachment(file);
          continue;
        }
        if (pendingFiles.length >= MAX_FILES) continue;
        const key = `${file.name}-${Date.now()}-${Math.random()}`;
        const mediaType = DOC_MIMES[file.name.split('.').pop()!.toLowerCase()];
        setPendingFiles((prev) =>
          prev.length >= MAX_FILES
            ? prev
            : [
                ...prev,
                {
                  key,
                  name: file.name,
                  size: file.size,
                  media_type: mediaType,
                  uploading: true,
                  progress: 0,
                },
              ],
        );
        const reader = new FileReader();
        reader.onload = () => {
          const base64 = String(reader.result || '').split(',')[1] ?? '';
          if (!base64) return;
          setPendingFiles((prev) =>
            prev.map((f) => (f.key === key ? { ...f, data: base64 } : f)),
          );
        };
        reader.readAsDataURL(file);
        uploadDurable(file, key);
      } else if (file.type.startsWith('text/') || TEXT_EXTENSIONS.test(file.name)) {
        if (file.size > MAX_FILE_CHARS) {
          // Too big to inline without truncating — store it whole instead.
          uploadAttachment(file);
          continue;
        }
        if (pendingFiles.length >= MAX_FILES) continue;
        const key = `${file.name}-${Date.now()}-${Math.random()}`;
        setPendingFiles((prev) =>
          prev.length >= MAX_FILES
            ? prev
            : [
                ...prev,
                {
                  key,
                  name: file.name,
                  size: file.size,
                  uploading: true,
                  progress: 0,
                },
              ],
        );
        const reader = new FileReader();
        reader.onload = () => {
          const text = String(reader.result || '').slice(0, MAX_FILE_CHARS);
          if (!text) return;
          setPendingFiles((prev) =>
            prev.map((f) =>
              f.key === key ? { ...f, content: text, size: text.length } : f,
            ),
          );
        };
        reader.readAsText(file);
        uploadDurable(file, key);
      } else {
        // Any other type (zip, sqlite, parquet, unknown binaries…): the
        // durable-upload rung — the agent works on it in the sandbox.
        uploadAttachment(file);
      }
    }
  }


  function openConversation(id: string) {
    navigate(`/chat/${id}`);
  }

  async function handleCreate() {
    try {
      const convo = await createConversation({});
      setConversations((prev) => [convo, ...prev]);
      navigate(`/chat/${convo.id}`);
    } catch {
      // handle silently
    }
  }

  async function handleDelete(id: string) {
    try {
      await deleteConversation(id);
      setConversations((prev) => prev.filter((c) => c.id !== id));
      if (activeId === id) {
        navigate('/chat');
      }
    } catch {
      // handle silently
    }
  }

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
          data.run_id &&
          prev.some((m) => m.metadata?.run_id === data.run_id)
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
        loadConversations();
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

  async function handleSend() {
    if (
      (!input.trim() && pendingImages.length === 0 && pendingFiles.length === 0) ||
      !activeId
    )
      return;
    if (pendingFiles.some((f) => f.uploading)) return; // wait for uploads
    const content = input.trim() || 'See the attached content.';

    // If the agent is waiting on a clarification, the message box answers it
    // (resumes the paused run) instead of starting a new turn.
    if (clarify) {
      setInput('');
      answerClarify(content);
      return;
    }
    if (sending) return;
    const images = pendingImages;
    const files = pendingFiles;
    setPendingImages([]);
    setPendingFiles([]);
    setInput('');
    await dispatchSend(content, images, files);
  }

  /** Re-send an earlier user message as a fresh turn (from its bubble). */
  async function handleResend(content: string) {
    if (!activeId || sending || clarify || !content.trim()) return;
    await dispatchSend(content.trim(), [], []);
  }

  /** Continue an interrupted response. If the stopped run's state is still
   *  warm on the server this truly resumes it (tool work intact); otherwise
   *  the continuation instruction runs as a normal turn over history. */
  async function handleContinueRun(message: MessageResponse) {
    if (!activeId || sending || clarify) return;
    const runId = message.metadata?.run_id ?? undefined;
    await dispatchSend(
      'Continue your interrupted response from exactly where it stopped. Do not repeat what you already wrote.',
      [],
      [],
      runId ? { resumeRunId: runId } : undefined,
    );
  }

  async function dispatchSend(
    content: string,
    images: MessageImage[],
    files: (MessageFile & { content?: string; data?: string; media_type?: string })[],
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

  /** Stop the in-flight run: the agent halts at the next step boundary and
   *  the stream finishes with a final frame (which resets the UI). */
  async function handleStop() {
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

  return (
    <div className="flex h-full flex-col">
      {/* Slim top bar: the conversation switcher lives HERE (a dropdown), so
          navigation costs zero horizontal space — the canvas is the chat's. */}
      <div className="flex h-12 shrink-0 items-center justify-between border-b border-zinc-800 px-2">
        <ConversationSwitcher
          conversations={conversations}
          activeId={activeId}
          onSelect={openConversation}
          onCreate={handleCreate}
          onDelete={handleDelete}
        />
        <Button
          variant="ghost"
          size="icon"
          onClick={handleCreate}
          title="New conversation"
        >
          <Plus size={16} />
        </Button>
      </div>

      {/* Chat area */}
      <div className="relative flex min-h-0 flex-1 flex-col">
        {!activeId ? (
          <div className="flex flex-1 flex-col items-center justify-center gap-4 text-zinc-500">
            <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-zinc-800">
              <Send size={24} />
            </div>
            <p className="text-lg font-medium">Select or start a conversation</p>
            <Button onClick={handleCreate}>New Conversation</Button>
          </div>
        ) : (
          <>
            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 lg:p-6">
              {loadingConversation ? (
                <div className="flex justify-center py-12">
                  <Spinner className="h-8 w-8" />
                </div>
              ) : messages.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-20 text-zinc-500">
                  <p className="text-sm">Send a message to start chatting</p>
                </div>
              ) : (
                <div className="mx-auto max-w-4xl space-y-6">
                  {messages.map((msg) => (
                    <MessageBubble
                      key={msg.id}
                      message={msg}
                      onOpenArtifact={setArtifactPath}
                      onResend={sending ? undefined : handleResend}
                      onContinue={sending ? undefined : handleContinueRun}
                    />
                  ))}
                  {streaming && streamText && (
                    <MessageBubble
                      message={{
                        id: 'streaming',
                        role: 'assistant',
                        content: streamText,
                        metadata: null,
                        created_at: new Date().toISOString(),
                      }}
                    />
                  )}
                  {streaming && (
                    <StreamingIndicator
                      status={streamStatus}
                      activity={streamActivity}
                      plan={streamPlan}
                      tools={streamTools}
                      step={streamStep}
                    />
                  )}
                  {clarify && (
                    <div className="mx-auto max-w-4xl rounded-xl border border-accent-500/40 bg-zinc-800 p-4">
                      <p className="mb-3 text-sm text-zinc-200">
                        {clarify.question}
                      </p>
                      {clarify.options.length > 0 ? (
                        <div className="flex flex-wrap gap-2">
                          {clarify.options.map((opt) => (
                            <button
                              key={opt}
                              type="button"
                              onClick={() => answerClarify(opt)}
                              className="rounded-full border border-zinc-600 px-3 py-1.5 text-sm text-zinc-200 hover:border-accent-500 hover:bg-accent-600/10 hover:text-accent-600"
                            >
                              {opt}
                            </button>
                          ))}
                        </div>
                      ) : (
                        <p className="text-xs text-zinc-500">
                          Type your answer in the message box below.
                        </p>
                      )}
                    </div>
                  )}
                  <div ref={messagesEndRef} />
                </div>
              )}
            </div>

            {/* Composer */}
            <div className="border-t border-zinc-800 p-4">
              <div className="mx-auto max-w-4xl">
                <Composer
                  value={input}
                  onChange={setInput}
                  onSend={handleSend}
                  onAddFiles={addAttachments}
                  pendingImages={pendingImages}
                  onRemoveImage={(i) =>
                    setPendingImages((prev) =>
                      prev.filter((_, idx) => idx !== i),
                    )
                  }
                  pendingFiles={pendingFiles}
                  onRemoveFile={(i) =>
                    setPendingFiles((prev) =>
                      prev.filter((_, idx) => idx !== i),
                    )
                  }
                  disabled={sending && !clarify}
                  placeholder={
                    clarify ? 'Answer the question above…' : 'Message Aloy…'
                  }
                  attachFull={
                    pendingImages.length >= MAX_IMAGES &&
                    pendingFiles.length >= MAX_FILES
                  }
                  onStop={sending && !clarify ? handleStop : undefined}
                />
              </div>
            </div>
          </>
        )}
      </div>

      {artifactPath && activeId && (
        <ArtifactDrawer
          conversationId={activeId}
          openPath={artifactPath}
          onClose={() => setArtifactPath(null)}
        />
      )}
    </div>
  );
}
