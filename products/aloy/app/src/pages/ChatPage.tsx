import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Plus, Send } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Spinner } from '@/components/ui/Spinner';
import { ConversationSwitcher } from '@/components/chat/ConversationSwitcher';
import { Composer } from '@/components/chat/Composer';
import { MessageList } from '@/components/chat/MessageList';
import { ArtifactDrawer } from '@/components/chat/ArtifactDrawer';
import { getConversation } from '@/api/conversations';
import { useConversations } from '@/hooks/useConversations';
import { useAttachments } from '@/hooks/useAttachments';
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
  const [artifactPath, setArtifactPath] = useState<string | null>(null);
  const [input, setInput] = useState('');
  const [loadingConversation, setLoadingConversation] = useState(false);

  const {
    pendingImages,
    pendingFiles,
    addAttachments,
    removeImage,
    removeFile,
    resetAttachments,
    uploadsInFlight,
    attachmentsFull,
  } = useAttachments(activeId);

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

            {/* Composer */}
            <div className="border-t border-zinc-800 p-4">
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
                  placeholder={
                    clarify
                      ? 'Answer the question above…'
                      : approval
                        ? 'Approve or reject the action above…'
                        : 'Message Aloy…'
                  }
                  attachFull={attachmentsFull}
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
