import { Check, Mail, X } from 'lucide-react';
import type { ApprovalDecision } from '@/api/sse';

/**
 * Inline approval card: the agent paused before a consequential action (sending
 * an email, etc.) and is waiting for the user to approve or reject it — the
 * commit step of a Proposal. Nothing is delivered until the user says yes.
 */
export function ApprovalPrompt({
  tool,
  args,
  onDecide,
}: {
  tool: string;
  args: Record<string, unknown>;
  onDecide: (decision: ApprovalDecision) => void;
}) {
  const isEmail = tool === 'gmail_send' || tool === 'gmail_send_draft';
  const title = isEmail ? 'Send this email?' : `Run ${tool}?`;

  return (
    <div className="mx-auto max-w-4xl rounded-xl border border-amber-500/40 bg-amber-500/5 p-4">
      <div className="mb-2 flex items-center gap-2 text-sm font-medium text-zinc-100">
        {isEmail ? (
          <Mail size={16} className="text-amber-500" />
        ) : (
          <Check size={16} className="text-amber-500" />
        )}
        {title}
        <span className="ml-auto rounded bg-amber-500/15 px-1.5 py-0.5 font-mono text-xs text-amber-600">
          needs approval
        </span>
      </div>

      <dl className="mb-3 space-y-1 rounded-lg border border-zinc-800 bg-zinc-900 p-3 text-sm">
        {fieldsFor(tool, args).map(([label, value]) => (
          <div key={label} className="flex gap-2">
            <dt className="w-16 shrink-0 text-xs font-medium text-zinc-500">
              {label}
            </dt>
            <dd className="whitespace-pre-wrap break-words text-zinc-200">{value}</dd>
          </div>
        ))}
      </dl>

      <div className="flex gap-2">
        <button
          type="button"
          onClick={() => onDecide({ type: 'approve' })}
          className="inline-flex items-center gap-1.5 rounded-lg bg-accent-600 px-3 py-1.5 text-sm font-medium text-white shadow-sm hover:bg-accent-500"
        >
          <Check size={15} /> {isEmail ? 'Send' : 'Approve'}
        </button>
        <button
          type="button"
          onClick={() => onDecide({ type: 'reject' })}
          className="inline-flex items-center gap-1.5 rounded-lg border border-zinc-700 px-3 py-1.5 text-sm font-medium text-zinc-300 hover:border-zinc-600 hover:text-zinc-100"
        >
          <X size={15} /> {isEmail ? "Don't send" : 'Reject'}
        </button>
      </div>
    </div>
  );
}

/** Human-friendly summary rows per tool, falling back to raw args for others. */
function fieldsFor(
  tool: string,
  args: Record<string, unknown>,
): [string, string][] {
  const str = (v: unknown) => (v == null ? '' : String(v));
  if (tool === 'gmail_send' || tool === 'gmail_send_draft') {
    // draft_id is enriched server-side into to/subject/body — show the email,
    // not the id. Fall back to the id only if enrichment couldn't read it.
    const rows: [string, string][] = [];
    if (args.to) rows.push(['To', str(args.to)]);
    if (args.subject) rows.push(['Subject', str(args.subject)]);
    if (args.cc) rows.push(['Cc', str(args.cc)]);
    if (args.body) rows.push(['Body', str(args.body)]);
    if (rows.length) return rows;
    if (args.draft_id) return [['Draft', str(args.draft_id)]];
  }
  return Object.entries(args).map(([k, v]) => [k, str(v)]);
}
