import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Plus, Send, X } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Spinner } from '@/components/ui/Spinner';
import { ConversationSwitcher } from '@/components/chat/ConversationSwitcher';
import { Composer } from '@/components/chat/Composer';
import { MessageList } from '@/components/chat/MessageList';
import { ArtifactDrawer } from '@/components/chat/ArtifactDrawer';
import { EventIcon } from '@/components/icons';
import { getConversation, getConversationMessages } from '@/api/conversations';
import { createEvent } from '@/api/events';
import { useConversations } from '@/hooks/useConversations';
import { useAttachments } from '@/hooks/useAttachments';
import { useFileReferences } from '@/hooks/useFileReferences';
import { useStreamingRun } from '@/hooks/useStreamingRun';
import type { MessageResponse } from '@/types';

/**
 * Thin composition over the chat hooks. All router coupling lives HERE —
 * route-change load/reset/reattach, redirect-to-most-recent, unmount abort —
 * while the hooks (useConversations / useAttachments / useStreamingRun) stay
 * page-agnostic so future Surfaces can reuse them.
 */
export function ChatPage() {
  const { conversationId: routeConversationId } = useParams();
  const navigate = useNavigate();

  const {
    conversations,
    activeId,
    setActiveId,
    loadConversations,
    createConversation,
    deleteConversation,
  } = useConversations();

  const [messages, setMessages] = useState<MessageResponse[]>([]);
  const [messageCursor, setMessageCursor] = useState<string | null>(null);
  const [loadingOlderMessages, setLoadingOlderMessages] = useState(false);
  const [artifactPath, setArtifactPath] = useState<string | null>(null);
  const [input, setInput] = useState('');
  const [loadingConversation, setLoadingConversation] = useState(false);
  const [creatingEvent, setCreatingEvent] = useState(false);
  const [eventTitle, setEventTitle] = useState('');
  const [eventSummary, setEventSummary] = useState('');
  const [savingEvent, setSavingEvent] = useState(false);
  const [eventError, setEventError] = useState('');

  const {
    pendingImages,
    pendingFiles,
    addAttachments,
    attachStoredFile,
    removeImage,
    removeFile,
    resetAttachments,
    uploadsInFlight,
    fileAttachmentsFull,
    attachmentsFull,
  } = useAttachments(activeId);
  const fileReferences = useFileReferences(activeId);

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
    activeId,
    setMessages,
    onConversationsRefresh: loadConversations,
  });

  useEffect(() => {
    loadConversations();
  }, [loadConversations]);

  useEffect(() => {
    // Kill any in-flight stream from the previous conversation and reset the
    // streaming UI + artifact drawer before rendering the new one.
    abortStream();
    resetStreamUi();
    // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional reset on route change
    setArtifactPath(null);

    let cancelled = false;
    if (routeConversationId) {
      setActiveId(routeConversationId);
      setLoadingConversation(true);
      getConversation(routeConversationId)
        .then((detail) => {
          if (cancelled) return;
          setMessages(detail.messages);
          setMessageCursor(detail.messages_next_cursor);
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
      setMessageCursor(null);
    }
    return () => {
      cancelled = true;
    };
    // tryReattach is deliberately NOT a dependency: it closes over per-render
    // callbacks, and this effect must re-run only on route change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [routeConversationId, abortStream, resetStreamUi, setActiveId]);

  // Abort the stream when leaving the chat page entirely.
  useEffect(() => {
    return () => abortStream();
  }, [abortStream]);

  // Landing on /chat with no conversation selected: open the most recent one
  // (nav 'Chat' should drop you into your chat, not an empty state).
  useEffect(() => {
    const mostRecent = conversations[0];
    if (!routeConversationId && mostRecent) {
      navigate(`/chat/${mostRecent.id}`, { replace: true });
    }
  }, [routeConversationId, conversations, navigate]);

  function openConversation(id: string) {
    navigate(`/chat/${id}`);
  }

  async function loadOlderMessages() {
    if (!activeId || !messageCursor || loadingOlderMessages) return;
    setLoadingOlderMessages(true);
    try {
      const page = await getConversationMessages(activeId, messageCursor);
      setMessages((current) => [...page.messages, ...current]);
      setMessageCursor(page.next_cursor);
    } finally {
      setLoadingOlderMessages(false);
    }
  }

  async function handleCreate() {
    const convo = await createConversation();
    if (convo) navigate(`/chat/${convo.id}`);
  }

  async function handleDelete(id: string) {
    const deleted = await deleteConversation(id);
    if (deleted && activeId === id) {
      navigate('/chat');
    }
  }

  async function handleCreateEvent() {
    if (!activeId || !eventTitle.trim()) return;
    setEventError('');
    setSavingEvent(true);
    try {
      const event = await createEvent({
        title: eventTitle.trim(),
        summary: eventSummary.trim(),
        phase: 'planning',
        origin_conversation_id: activeId,
      });
      navigate(`/events/${event.id}`);
    } catch (cause) {
      setEventError(cause instanceof Error ? cause.message : 'Could not create the Event');
    } finally {
      setSavingEvent(false);
    }
  }

  async function handleSend() {
    if (
      (!input.trim() && pendingImages.length === 0 && pendingFiles.length === 0) ||
      !activeId
    )
      return;
    if (uploadsInFlight) return; // wait for uploads
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
    resetAttachments();
    setInput('');
    await dispatchSend(content, images, files);
  }

  return (
    <div className="flex h-full flex-col">
      {/* Slim top bar: the conversation switcher lives HERE (a dropdown), so
          navigation costs zero horizontal space — the canvas is the chat's. */}
      <div className="flex min-h-12 shrink-0 items-center justify-between gap-1 border-b border-zinc-800 px-1.5 sm:px-2">
        <ConversationSwitcher
          conversations={conversations}
          activeId={activeId}
          onSelect={openConversation}
          onCreate={handleCreate}
          onDelete={handleDelete}
        />
        <div className="flex items-center gap-1">
          <Button
            variant="outline"
            size="sm"
            className="border-zinc-700 bg-zinc-900 text-zinc-300 hover:border-accent-500/45 hover:bg-zinc-800 hover:text-accent-200"
            onClick={() => setCreatingEvent((value) => !value)}
            title="Create Event from this conversation"
            disabled={!activeId}
          >
            <EventIcon size={16} />
            <span className="hidden sm:inline">Create Event</span>
          </Button>
          <Button
            variant="ghost"
            size="icon"
            onClick={handleCreate}
            title="New conversation"
          >
            <Plus size={16} />
          </Button>
        </div>
      </div>

      {creatingEvent && activeId && (
        <div className="border-b border-zinc-800 bg-zinc-900 px-3 py-4 sm:px-4">
          <div className="mx-auto max-w-4xl">
            <div className="mb-3 flex items-center gap-2">
              <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-accent-500/10 text-accent-300">
                <EventIcon size={15} />
              </span>
              <div>
                <p className="text-sm font-semibold text-zinc-100">Turn this into an Event</p>
                <p className="text-xs text-zinc-500">Give this work a durable workspace and continuous session.</p>
              </div>
            </div>
            <div className="flex flex-col gap-2 sm:flex-row">
              <input
                value={eventTitle}
                onChange={(event) => setEventTitle(event.target.value)}
                placeholder="Event name"
                autoFocus
                className="min-w-0 flex-1 rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 placeholder-zinc-500 focus:border-accent-500 focus:outline-none"
              />
              <input
                value={eventSummary}
                onChange={(event) => setEventSummary(event.target.value)}
                placeholder="What does success look like?"
                className="min-w-0 flex-[1.5] rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 placeholder-zinc-500 focus:border-accent-500 focus:outline-none"
              />
              <Button
                disabled={savingEvent || !eventTitle.trim()}
                onClick={handleCreateEvent}
              >
                {savingEvent ? 'Creating…' : 'Create Event'}
              </Button>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setCreatingEvent(false)}
                title="Cancel"
              >
                <X size={16} />
              </Button>
            </div>
            {eventError && <p className="mt-2 text-xs text-red-400">{eventError}</p>}
          </div>
        </div>
      )}

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
            <div className="flex-1 overflow-y-auto px-4 py-8 sm:px-6 sm:py-10 lg:px-8 lg:py-12">
              {loadingConversation ? (
                <div className="flex justify-center py-12">
                  <Spinner className="h-8 w-8" />
                </div>
              ) : messages.length === 0 ? (
                <div className="flex min-h-[55vh] flex-col items-center justify-center text-center">
                  <h2 className="font-display text-2xl font-semibold text-zinc-200">What would you like to work on?</h2>
                  <p className="mt-2 text-sm text-zinc-500">Aloy can think with you, organize work, or help you begin an Event.</p>
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
                  hasOlder={!!messageCursor}
                  loadingOlder={loadingOlderMessages}
                  onLoadOlder={() => void loadOlderMessages()}
                />
              )}
            </div>

            {/* Composer */}
            <div className="shrink-0 bg-zinc-950/95 px-3 pb-3 pt-2 backdrop-blur sm:px-6 sm:pb-5 lg:px-8">
              <div className="mx-auto max-w-[56rem]">
                <Composer
                  value={input}
                  onChange={setInput}
                  onSend={handleSend}
                  onAddFiles={addAttachments}
                  onChooseFile={attachStoredFile}
                  onSearchFiles={fileReferences.search}
                  referenceFiles={fileReferences.files}
                  referenceFilesLoading={fileReferences.loading}
                  referenceFilesError={fileReferences.error}
                  referenceScopeLabel="All files you own across Life and Events"
                  pendingImages={pendingImages}
                  onRemoveImage={removeImage}
                  pendingFiles={pendingFiles}
                  onRemoveFile={removeFile}
                  disabled={sending && !clarify}
                  placeholder={
                    clarify
                      ? 'Answer the question above…'
                      : approval
                        ? 'Approve or reject the action above…'
                        : 'Ask Aloy anything…'
                  }
                  attachFull={attachmentsFull}
                  fileAttachFull={fileAttachmentsFull}
                  onStop={sending && !clarify ? stopRun : undefined}
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
