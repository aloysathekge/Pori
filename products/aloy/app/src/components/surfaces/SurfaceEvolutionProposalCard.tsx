import { useState } from 'react';
import { ArrowUpRight, Wrench } from 'lucide-react';
import {
  decideSurfaceEvolutionProposal,
  type SurfaceEvolutionProposal,
} from '@/api/surfaces';
import { Button } from '@/components/ui/Button';

interface SurfaceEvolutionProposalCardProps {
  eventId: string;
  proposal: SurfaceEvolutionProposal;
  onDecided: (proposal: SurfaceEvolutionProposal) => void;
}

function evidenceLabel(proposal: SurfaceEvolutionProposal) {
  if (proposal.trigger === 'primary_job_failure') {
    return `Observed ${proposal.occurrence_count} times`;
  }
  if (proposal.trigger === 'event_phase_changed') return 'Your Event phase changed';
  if (proposal.trigger === 'negative_feedback') return 'Based on your feedback';
  return 'Based on trusted Surface evidence';
}

export function SurfaceEvolutionProposalCard({
  eventId,
  proposal,
  onDecided,
}: SurfaceEvolutionProposalCardProps) {
  const [acting, setActing] = useState<'accept' | 'dismiss' | null>(null);
  const [error, setError] = useState('');

  async function decide(decision: 'accept' | 'dismiss') {
    setActing(decision);
    setError('');
    try {
      onDecided(
        await decideSurfaceEvolutionProposal(eventId, proposal.id, decision),
      );
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setActing(null);
    }
  }

  return (
    <section className="mx-auto mt-4 max-w-xl rounded-2xl border border-zinc-700/80 bg-zinc-900/75 p-4 shadow-lg shadow-black/10">
      <div className="flex items-start gap-3">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border border-zinc-700 bg-zinc-950 text-accent-600">
          <Wrench size={16} />
        </div>
        <div className="min-w-0 flex-1">
          <p className="text-xs font-medium text-zinc-500">Surface improvement</p>
          <h3 className="mt-1 text-sm font-semibold leading-5 text-zinc-100">
            {proposal.goal}
          </h3>
          <p className="mt-1.5 text-xs leading-5 text-zinc-500">
            {evidenceLabel(proposal)}. Aloy will keep the current Surface live while
            a replacement is built and checked.
          </p>
          {error && <p role="alert" className="mt-2 text-xs text-red-400">{error}</p>}
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <Button
              size="sm"
              onClick={() => void decide('accept')}
              disabled={acting !== null}
            >
              {acting === 'accept' ? 'Starting…' : 'Improve Surface'}
              <ArrowUpRight size={14} />
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => void decide('dismiss')}
              disabled={acting !== null}
            >
              {acting === 'dismiss' ? 'Dismissing…' : 'Not now'}
            </Button>
          </div>
        </div>
      </div>
    </section>
  );
}
