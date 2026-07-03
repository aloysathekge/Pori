import { useState } from "react";
import { ChevronRight, Wrench } from "lucide-react";
import type { ChatMessage } from "@/types";

export function Message({ message }: { message: ChatMessage }) {
  const [showThinking, setShowThinking] = useState(false);
  const isUser = message.role === "user";

  return (
    <div className={isUser ? "max-w-full self-end" : "max-w-full self-start"}>
      <div
        className={
          isUser
            ? "rounded-2xl bg-aloy-accent-soft px-4 py-3"
            : "rounded-2xl border border-aloy-border bg-aloy-panel px-4 py-3"
        }
      >
        {message.thinking && (
          <div className="mb-2">
            <button
              type="button"
              onClick={() => setShowThinking((s) => !s)}
              className="flex items-center gap-1 text-xs text-aloy-muted hover:text-aloy-text"
            >
              <ChevronRight
                className={`h-3 w-3 transition-transform ${showThinking ? "rotate-90" : ""}`}
              />
              thinking
            </button>
            {showThinking && (
              <pre className="mt-1 whitespace-pre-wrap text-xs text-aloy-muted">
                {message.thinking}
              </pre>
            )}
          </div>
        )}

        {message.tools.length > 0 && (
          <div className="mb-2 flex flex-wrap gap-1">
            {message.tools.map((t, i) => (
              <span
                key={i}
                className="inline-flex items-center gap-1 rounded-full border border-aloy-border px-2 py-0.5 text-xs text-aloy-muted"
              >
                <Wrench className="h-3 w-3" />
                {t.name}
                {t.success === false ? " ✗" : t.success ? " ✓" : " …"}
              </span>
            ))}
          </div>
        )}

        <div className="whitespace-pre-wrap text-sm leading-relaxed">
          {message.text ||
            (message.streaming ? <span className="text-aloy-muted">…</span> : "")}
        </div>
      </div>
    </div>
  );
}
