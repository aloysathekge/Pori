import { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Send } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Spinner } from '@/components/ui/Spinner';
import { ConversationList } from '@/components/chat/ConversationList';
import { MessageBubble } from '@/components/chat/MessageBubble';
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
  MessageResponse,
  SSEMessageEvent,
  SSEToolEvent,
  PlanItem,
} from '@/types';

export function ChatPage() {
  const { conversationId: routeConversationId } = useParams();
  const navigate = useNavigate();
  const [conversations, setConversations] = useState<ConversationResponse[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<MessageResponse[]>([]);
  const [input, setInput] = useState('');
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
    if (routeConversationId) {
      selectConversation(routeConversationId);
    } else {
      setActiveId(null);
      setMessages([]);
    }
  }, [routeConversationId]);

  async function loadConversations() {
    try {
      const convos = await listConversations(50);
      setConversations(convos);
    } catch {
      // handle silently
    }
  }

  async function selectConversation(id: string) {
    setActiveId(id);
    setLoadingConversation(true);
    try {
      const detail = await getConversation(id);
      setMessages(detail.messages);
    } catch {
      setMessages([]);
    } finally {
      setLoadingConversation(false);
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
    if (!input.trim() || !activeId || sending) return;

    const content = input.trim();
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
      metadata: null,
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
        setStreamText('');
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
        setStreamText('');
      },
      onDone: () => {
        setStreaming(false);
        setSending(false);
        setStreamStatus('');
        setStreamActivity('');
        setStreamPlan([]);
        setStreamTools([]);
        setStreamStep(undefined);
        setStreamText('');
        setClarify(null);
        loadConversations();
      },
    };

    try {
      await streamMessage(activeId, content, callbacks);
    } catch {
      setStreaming(false);
      setSending(false);
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
    <div className="flex h-full">
      {/* Sidebar */}
      <div className="hidden w-72 lg:block">
        <ConversationList
          conversations={conversations}
          activeId={activeId}
          onSelect={openConversation}
          onCreate={handleCreate}
          onDelete={handleDelete}
        />
      </div>

      {/* Chat area */}
      <div className="flex flex-1 flex-col">
        {!activeId ? (
          <div className="flex flex-1 flex-col items-center justify-center gap-4 text-zinc-500">
            <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-zinc-800">
              <Send size={24} />
            </div>
            <p className="text-lg font-medium">Select or start a conversation</p>
            <Button onClick={handleCreate}>New Conversation</Button>
            {/* Mobile conversation list */}
            <div className="mt-4 w-full max-w-sm lg:hidden">
              <ConversationList
                conversations={conversations}
                activeId={activeId}
                onSelect={openConversation}
                onCreate={handleCreate}
                onDelete={handleDelete}
              />
            </div>
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
                <div className="mx-auto max-w-3xl space-y-6">
                  {messages.map((msg) => (
                    <MessageBubble key={msg.id} message={msg} />
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
                    <div className="mx-auto max-w-3xl rounded-xl border border-zinc-700 bg-zinc-800 p-4">
                      <p className="mb-3 text-sm text-zinc-200">
                        {clarify.question}
                      </p>
                      <div className="flex flex-wrap gap-2">
                        {clarify.options.map((opt) => (
                          <button
                            key={opt}
                            type="button"
                            onClick={() => answerClarify(opt)}
                            className="rounded-full border border-zinc-600 px-3 py-1 text-sm text-zinc-200 hover:border-accent-500 hover:text-accent-700"
                          >
                            {opt}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                  <div ref={messagesEndRef} />
                </div>
              )}
            </div>

            {/* Input bar */}
            <div className="border-t border-zinc-800 p-4">
              <div className="mx-auto flex max-w-3xl gap-3">
                <input
                  className="flex-1 rounded-xl border border-zinc-700 bg-zinc-800 px-4 py-3 text-sm text-zinc-100 placeholder-zinc-500 focus:border-accent-500 focus:outline-none focus:ring-1 focus:ring-accent-500"
                  placeholder="Type a message..."
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSend();
                    }
                  }}
                  disabled={sending}
                />
                <Button
                  onClick={handleSend}
                  disabled={!input.trim() || sending}
                  size="icon"
                  className="h-12 w-12 rounded-xl"
                >
                  <Send size={18} />
                </Button>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
