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
} from '@/api/conversations';
import {
  streamMessage,
  submitClarification,
  type SSECallbacks,
} from '@/api/sse';
import type {
  ConversationResponse,
  MessageImage,
  MessageResponse,
  SSEMessageEvent,
  SSEToolEvent,
  PlanItem,
} from '@/types';

const MAX_IMAGES = 3;
const MAX_IMAGE_BYTES = 5 * 1024 * 1024; // 5MB per image (backend-enforced too)

export function ChatPage() {
  const { conversationId: routeConversationId } = useParams();
  const navigate = useNavigate();
  const [conversations, setConversations] = useState<ConversationResponse[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<MessageResponse[]>([]);
  const [artifactPath, setArtifactPath] = useState<string | null>(null);
  const [input, setInput] = useState('');
  const [pendingImages, setPendingImages] = useState<MessageImage[]>([]);
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
          if (!cancelled) setMessages(detail.messages);
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
  }, [routeConversationId, resetStreamUi]);

  // Abort the stream when leaving the chat page entirely.
  useEffect(() => {
    return () => streamAbortRef.current?.abort();
  }, []);

  // Landing on /chat with no conversation selected: open the most recent one
  // (nav 'Chat' should drop you into your chat, not an empty state).
  useEffect(() => {
    if (!routeConversationId && conversations.length > 0) {
      navigate(`/chat/${conversations[0].id}`, { replace: true });
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

  /** Read image files into base64 attachments (attach button, paste, drop). */
  function addImageFiles(files: Iterable<File>) {
    for (const file of files) {
      if (!file.type.startsWith('image/')) continue;
      if (file.size > MAX_IMAGE_BYTES) continue; // silently skip oversized
      const reader = new FileReader();
      reader.onload = () => {
        const url = String(reader.result || '');
        const base64 = url.split(',')[1] ?? '';
        if (!base64) return;
        setPendingImages((prev) =>
          prev.length >= MAX_IMAGES
            ? prev
            : [...prev, { data: base64, media_type: file.type }],
        );
      };
      reader.readAsDataURL(file);
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

  async function handleSend() {
    if ((!input.trim() && pendingImages.length === 0) || !activeId) return;
    const content = input.trim() || 'See the attached image.';

    // If the agent is waiting on a clarification, the message box answers it
    // (resumes the paused run) instead of starting a new turn.
    if (clarify) {
      setInput('');
      answerClarify(content);
      return;
    }
    if (sending) return;
    const images = pendingImages;
    setPendingImages([]);
    setInput('');
    setSending(true);
    setStreaming(true);
    setStreamStatus('Starting...');
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
      metadata: images.length > 0 ? { images } : null,
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, userMsg]);

    const callbacks: SSECallbacks = {
      onText: (text) => setStreamText((prev) => prev + text),
      onToolStart: (payload) =>
        setStreamTools((prev) => [
          ...prev,
          {
            step: 0,
            tool: String(payload.name ?? 'tool'),
            preview: '',
            success: false,
          },
        ]),
      onToolEnd: (payload) =>
        setStreamTools((prev) => {
          const next = [...prev];
          const name = String(payload.name ?? '');
          for (let i = next.length - 1; i >= 0; i--) {
            if (next[i].tool === name) {
              next[i] = { ...next[i], success: Boolean(payload.success) };
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
          },
          created_at: new Date().toISOString(),
        };
        setMessages((prev) => [...prev, assistantMsg]);
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

    // Abortable stream: switching conversations or leaving the page aborts it
    // so tokens can't bleed into another conversation's view.
    const controller = new AbortController();
    streamAbortRef.current = controller;
    try {
      await streamMessage(activeId, content, callbacks, {
        images: images.length > 0 ? images : undefined,
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
                  onAddImages={addImageFiles}
                  pendingImages={pendingImages}
                  onRemoveImage={(i) =>
                    setPendingImages((prev) =>
                      prev.filter((_, idx) => idx !== i),
                    )
                  }
                  disabled={sending && !clarify}
                  placeholder={
                    clarify ? 'Answer the question above…' : 'Message Aloy…'
                  }
                  maxImages={MAX_IMAGES}
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
