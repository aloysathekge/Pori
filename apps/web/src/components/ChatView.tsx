import { useCallback, useState } from "react";
import type { ClarificationRequestPayload } from "@aloy/shared";
import { client } from "@/lib/client";
import type { ChatMessage, PendingClarification } from "@/types";
import { Message } from "./Message";
import { Composer } from "./Composer";
import { ClarifyButtons } from "./ClarifyButtons";

let counter = 0;
const nextId = () => `m${++counter}`;

export function ChatView() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [clarify, setClarify] = useState<PendingClarification | null>(null);
  const [busy, setBusy] = useState(false);

  const patch = useCallback(
    (id: string, fn: (m: ChatMessage) => ChatMessage) =>
      setMessages((prev) => prev.map((m) => (m.id === id ? fn(m) : m))),
    [],
  );

  const send = useCallback(
    async (text: string) => {
      if (!text.trim() || busy) return;
      const user: ChatMessage = {
        id: nextId(),
        role: "user",
        text,
        thinking: "",
        tools: [],
        streaming: false,
      };
      const id = nextId();
      const assistant: ChatMessage = {
        id,
        role: "assistant",
        text: "",
        thinking: "",
        tools: [],
        streaming: true,
      };
      setMessages((prev) => [...prev, user, assistant]);
      setBusy(true);
      try {
        await client.streamTask(text, {
          onText: (chunk) => patch(id, (m) => ({ ...m, text: m.text + chunk })),
          onThinking: (chunk) =>
            patch(id, (m) => ({ ...m, thinking: m.thinking + chunk })),
          onToolStart: (e) =>
            patch(id, (m) => ({
              ...m,
              tools: [...m.tools, { name: toolName(e.payload) }],
            })),
          onToolEnd: (e) =>
            patch(id, (m) => ({ ...m, tools: markToolDone(m.tools, e.payload) })),
          onClarification: (req: ClarificationRequestPayload) =>
            setClarify({ id: req.id, question: req.question, options: req.options }),
          onError: (err) =>
            patch(id, (m) => ({ ...m, text: `${m.text}\n\n[error: ${String(err)}]` })),
        });
      } finally {
        patch(id, (m) => ({ ...m, streaming: false }));
        setBusy(false);
      }
    },
    [busy, patch],
  );

  const answerClarify = useCallback(
    async (value: string) => {
      if (!clarify) return;
      const id = clarify.id;
      setClarify(null);
      await client.submitClarification(id, value).catch(() => undefined);
    },
    [clarify],
  );

  return (
    <div className="mx-auto flex h-full max-w-3xl flex-col">
      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-6">
        {messages.length === 0 && (
          <p className="mt-10 text-center text-aloy-muted">Ask Aloy anything.</p>
        )}
        <div className="flex flex-col gap-4">
          {messages.map((m) => (
            <Message key={m.id} message={m} />
          ))}
        </div>
        {clarify && <ClarifyButtons clarify={clarify} onAnswer={answerClarify} />}
      </div>
      <Composer disabled={busy} onSend={send} />
    </div>
  );
}

function toolName(payload: Record<string, unknown>): string {
  return typeof payload.name === "string" ? payload.name : "tool";
}

function markToolDone(
  tools: ChatMessage["tools"],
  payload: Record<string, unknown>,
): ChatMessage["tools"] {
  const name = typeof payload.name === "string" ? payload.name : "";
  const success = Boolean(payload.success);
  const next = [...tools];
  const idx = next.map((t) => t.name).lastIndexOf(name);
  if (idx >= 0) next[idx] = { ...next[idx], success };
  return next;
}
