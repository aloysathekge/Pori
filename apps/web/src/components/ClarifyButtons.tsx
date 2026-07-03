import type { PendingClarification } from "@/types";

interface ClarifyButtonsProps {
  clarify: PendingClarification;
  onAnswer: (value: string) => void;
}

/** Renders a structured `ask_user` request as buttons (the clarify bridge). */
export function ClarifyButtons({ clarify, onAnswer }: ClarifyButtonsProps) {
  return (
    <div className="mt-4 rounded-2xl border border-aloy-border bg-aloy-panel p-4">
      <p className="mb-3 text-sm">{clarify.question}</p>
      <div className="flex flex-wrap gap-2">
        {clarify.options.map((option) => (
          <button
            key={option}
            type="button"
            onClick={() => onAnswer(option)}
            className="rounded-full border border-aloy-border px-3 py-1 text-sm hover:border-aloy-accent hover:text-aloy-accent"
          >
            {option}
          </button>
        ))}
      </div>
    </div>
  );
}
