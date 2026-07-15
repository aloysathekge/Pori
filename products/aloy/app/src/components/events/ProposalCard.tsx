import { useState } from 'react';
import { Check, ShieldAlert, X } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import type { EventProposal } from '@/api/events';

interface ProposalCardProps {
  proposal: EventProposal;
  onDecision: (decision: 'approve' | 'reject') => Promise<void>;
}

export function ProposalCard({ proposal, onDecision }: ProposalCardProps) {
  const [busy, setBusy] = useState<'approve' | 'reject' | null>(null);

  async function decide(decision: 'approve' | 'reject') {
    setBusy(decision);
    try {
      await onDecision(decision);
    } finally {
      setBusy(null);
    }
  }

  return (
    <article className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-4">
      <div className="flex items-start gap-3">
        <span className="mt-0.5 rounded-lg bg-amber-500/15 p-2 text-amber-600">
          <ShieldAlert size={17} />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="font-medium text-zinc-100">{proposal.reason}</h3>
            <span className="rounded-full border border-zinc-700 px-2 py-0.5 font-mono text-[11px] text-zinc-400">
              {proposal.tool}
            </span>
          </div>
          <p className="mt-1 text-sm text-zinc-400">{proposal.impact}</p>
          <details className="mt-3 text-xs text-zinc-500">
            <summary className="cursor-pointer select-none hover:text-zinc-300">
              Review exact action
            </summary>
            <pre className="mt-2 overflow-x-auto rounded-lg bg-zinc-950/70 p-3 text-zinc-400">
              {JSON.stringify(proposal.args, null, 2)}
            </pre>
          </details>
          <div className="mt-4 flex gap-2">
            <Button
              size="sm"
              onClick={() => decide('approve')}
              disabled={busy !== null}
            >
              <Check size={15} /> {busy === 'approve' ? 'Approving…' : 'Approve'}
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={() => decide('reject')}
              disabled={busy !== null}
            >
              <X size={15} /> {busy === 'reject' ? 'Rejecting…' : 'Reject'}
            </Button>
          </div>
        </div>
      </div>
    </article>
  );
}
