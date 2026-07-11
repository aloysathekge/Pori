/**
 * Inline clarification panel: the agent paused mid-run to ask the user a
 * question. Options render as one-tap chips; with no options the user
 * answers through the message box instead.
 */
export function ClarifyPrompt({
  question,
  options,
  onAnswer,
}: {
  question: string;
  options: string[];
  onAnswer: (value: string) => void;
}) {
  return (
    <div className="mx-auto max-w-4xl rounded-xl border border-accent-500/40 bg-zinc-800 p-4">
      <p className="mb-3 text-sm text-zinc-200">{question}</p>
      {options.length > 0 ? (
        <div className="flex flex-wrap gap-2">
          {options.map((opt) => (
            <button
              key={opt}
              type="button"
              onClick={() => onAnswer(opt)}
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
  );
}
