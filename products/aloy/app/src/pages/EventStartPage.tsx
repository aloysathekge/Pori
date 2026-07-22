import { useEffect, useMemo, useRef, useState, type ComponentType } from 'react';
import {
  ArrowLeft,
  ArrowRight,
  BriefcaseBusiness,
  Building2,
  Check,
  GraduationCap,
  LayoutTemplate,
  LoaderCircle,
  RefreshCw,
  UserRound,
  Users,
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import {
  getEventTemplate,
  installEventTemplate,
  listEventTemplates,
  type EventTemplateDetail,
  type EventTemplateDiscoveryGroup,
  type EventTemplateSummary,
} from '@/api/eventTemplates';
import { AloyMark, EventIcon } from '@/components/icons';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Modal } from '@/components/ui/Modal';

type GroupPresentation = {
  label: string;
  Icon: ComponentType<{ size?: number; className?: string }>;
  tint: string;
};

const GROUPS: Record<string, GroupPresentation> = {
  student: { label: 'Student', Icon: GraduationCap, tint: 'bg-sky-500/10 text-sky-400' },
  individual: { label: 'Individual', Icon: UserRound, tint: 'bg-violet-500/10 text-violet-400' },
  professional: { label: 'Professional', Icon: BriefcaseBusiness, tint: 'bg-amber-500/10 text-amber-400' },
  team: { label: 'Team', Icon: Users, tint: 'bg-emerald-500/10 text-emerald-400' },
  business: { label: 'Business', Icon: Building2, tint: 'bg-rose-500/10 text-rose-400' },
};

function groupPresentation(group: EventTemplateDiscoveryGroup): GroupPresentation {
  return GROUPS[group] ?? {
    label: group.charAt(0).toUpperCase() + group.slice(1),
    Icon: LayoutTemplate,
    tint: 'bg-zinc-800 text-zinc-300',
  };
}

function installationKey(templateId: string) {
  return `event-template:${templateId}:${crypto.randomUUID()}`;
}

function TemplateCard({
  template,
  onOpen,
}: {
  template: EventTemplateSummary;
  onOpen: (template: EventTemplateSummary) => void;
}) {
  const group = groupPresentation(template.discovery_group);
  const GroupIcon = group.Icon;
  return (
    <button
      type="button"
      onClick={() => onOpen(template)}
      className="group flex min-h-60 w-full flex-col overflow-hidden rounded-3xl border border-zinc-800 bg-zinc-900 text-left transition hover:-translate-y-0.5 hover:border-accent-600/45 hover:bg-zinc-900/80 hover:shadow-xl hover:shadow-black/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-500"
    >
      <div className="relative flex min-h-28 items-end overflow-hidden border-b border-zinc-800 bg-[radial-gradient(circle_at_20%_20%,rgba(15,133,113,0.22),transparent_38%),linear-gradient(135deg,rgba(39,39,42,0.9),rgba(9,9,11,1))] p-5">
        <span className={`flex h-11 w-11 items-center justify-center rounded-2xl ${group.tint}`}>
          <GroupIcon size={21} />
        </span>
        <span className="absolute right-5 top-5 rounded-full border border-white/10 bg-black/20 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.12em] text-zinc-400">
          v{template.current_release.version}
        </span>
      </div>
      <div className="flex flex-1 flex-col p-5">
        <p className="text-xs font-semibold uppercase tracking-[0.12em] text-zinc-500">{group.label}</p>
        <h2 className="mt-2 font-display text-xl font-semibold text-zinc-100">{template.title}</h2>
        <p className="mt-2 line-clamp-3 text-sm leading-6 text-zinc-400">{template.summary}</p>
        <span className="mt-auto flex items-center gap-2 pt-5 text-sm font-semibold text-accent-700">
          See what it sets up <ArrowRight size={15} className="transition-transform group-hover:translate-x-0.5" />
        </span>
      </div>
    </button>
  );
}

export function EventStartPage() {
  const navigate = useNavigate();
  const [templates, setTemplates] = useState<EventTemplateSummary[]>([]);
  const [selected, setSelected] = useState<EventTemplateSummary | null>(null);
  const [detail, setDetail] = useState<EventTemplateDetail | null>(null);
  const [title, setTitle] = useState('');
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [installing, setInstalling] = useState(false);
  const [catalogError, setCatalogError] = useState('');
  const [installError, setInstallError] = useState('');
  const idempotencyKey = useRef<string | null>(null);

  async function loadCatalog() {
    setLoading(true);
    setCatalogError('');
    try {
      const response = await listEventTemplates();
      setTemplates(response.templates);
    } catch (cause) {
      setCatalogError(cause instanceof Error ? cause.message : 'Could not load starting templates');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    let active = true;
    void listEventTemplates()
      .then((response) => {
        if (active) setTemplates(response.templates);
      })
      .catch((cause: unknown) => {
        if (active) {
          setCatalogError(cause instanceof Error ? cause.message : 'Could not load starting templates');
        }
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  const groupedTemplates = useMemo(() => {
    const groups = new Map<string, EventTemplateSummary[]>();
    for (const template of templates) {
      const values = groups.get(template.discovery_group) ?? [];
      values.push(template);
      groups.set(template.discovery_group, values);
    }
    return Array.from(groups.entries());
  }, [templates]);

  async function openTemplate(template: EventTemplateSummary) {
    setSelected(template);
    setTitle(template.title);
    setDetail(null);
    setInstallError('');
    setDetailLoading(true);
    idempotencyKey.current = null;
    try {
      setDetail(await getEventTemplate(template.id));
    } catch {
      // The published catalog card is enough to install. Extra release detail
      // improves the preview but must not turn template discovery into a trap.
    } finally {
      setDetailLoading(false);
    }
  }

  function closeTemplate() {
    if (installing) return;
    setSelected(null);
    setDetail(null);
    setInstallError('');
    idempotencyKey.current = null;
  }

  async function installSelected() {
    if (!selected || !title.trim()) return;
    setInstalling(true);
    setInstallError('');
    idempotencyKey.current ??= installationKey(selected.id);
    try {
      const result = await installEventTemplate(selected.id, {
        idempotency_key: idempotencyKey.current,
        release_id: selected.current_release.id,
        title: title.trim(),
      });
      navigate(`/events/${result.event.id}`, { replace: true });
    } catch (cause) {
      setInstallError(cause instanceof Error ? cause.message : 'Could not start this Event');
      setInstalling(false);
    }
  }

  return (
    <div className="h-full overflow-y-auto bg-zinc-950">
      <div className="mx-auto min-h-full max-w-6xl px-5 py-5 sm:px-8 lg:px-10 lg:py-8">
        <header className="flex items-center justify-between gap-3">
          <button type="button" onClick={() => navigate('/today')} className="flex min-h-11 items-center gap-2 rounded-lg text-sm text-zinc-500 hover:text-zinc-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-500">
            <ArrowLeft size={16} /> Today
          </button>
          <Button variant="outline" onClick={() => navigate('/events/new')}>
            <EventIcon size={17} /> Start simple
          </Button>
        </header>

        <main className="pb-10 pt-7 sm:pt-10">
          <div className="max-w-3xl">
            <div className="flex items-center gap-2 text-accent-600"><AloyMark size={23} /><span className="text-sm font-semibold">Start an Event</span></div>
            <h1 className="mt-3 font-display text-3xl font-semibold tracking-tight text-zinc-100 sm:text-5xl">
              Begin with a useful head start.
            </h1>
            <p className="mt-4 max-w-2xl text-sm leading-6 text-zinc-400 sm:text-base">
              Pick a starting setup for the kind of work you are taking on, or create your own. Every template becomes your ordinary Event and can change with you over time.
            </p>
          </div>

          {loading ? (
            <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-3" aria-label="Loading Event templates">
              {[0, 1, 2].map((item) => <div key={item} className="h-60 animate-pulse rounded-3xl border border-zinc-800 bg-zinc-900" />)}
            </div>
          ) : catalogError ? (
            <div className="mt-10 flex flex-col items-start gap-4 rounded-3xl border border-zinc-800 bg-zinc-900 p-6 sm:flex-row sm:items-center sm:justify-between">
              <div><h2 className="font-display text-lg font-semibold text-zinc-100">Templates are temporarily unavailable</h2><p className="mt-1 text-sm text-zinc-500">You can still start a custom Event now.</p></div>
              <Button variant="secondary" onClick={() => void loadCatalog()}><RefreshCw size={16} /> Try again</Button>
            </div>
          ) : groupedTemplates.length > 0 ? (
            <div className="mt-10 space-y-9">
              {groupedTemplates.map(([groupName, values]) => {
                const group = groupPresentation(groupName);
                return (
                  <section key={groupName} aria-labelledby={`template-group-${groupName}`}>
                    <div className="mb-3 flex items-baseline justify-between gap-3">
                      <h2 id={`template-group-${groupName}`} className="font-display text-lg font-semibold text-zinc-100">{group.label}</h2>
                      <span className="text-xs text-zinc-600">{values.length} {values.length === 1 ? 'setup' : 'setups'}</span>
                    </div>
                    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                      {values.map((template) => <TemplateCard key={template.id} template={template} onOpen={(value) => void openTemplate(value)} />)}
                    </div>
                  </section>
                );
              })}
            </div>
          ) : (
            <div className="mt-10 rounded-3xl border border-dashed border-zinc-800 px-6 py-10 text-center">
              <LayoutTemplate className="mx-auto text-zinc-600" size={26} />
              <h2 className="mt-3 font-display text-lg font-semibold text-zinc-200">Starting templates are coming here</h2>
              <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-zinc-500">For now, create your Event from its name and the context you already have.</p>
            </div>
          )}

          <section className="mt-10 flex flex-col gap-5 border-t border-zinc-800 pt-7 sm:flex-row sm:items-center sm:justify-between">
            <div><h2 className="font-display text-lg font-semibold text-zinc-100">Have something different in mind?</h2><p className="mt-1 text-sm text-zinc-500">Name it, add context, files, links, or connections, and let the Event grow from there.</p></div>
            <Button onClick={() => navigate('/events/new')} className="shrink-0"><EventIcon size={17} /> Create your own Event</Button>
          </section>
        </main>
      </div>

      <Modal open={selected !== null} onClose={closeTemplate} title={selected?.title ?? 'Event template'} panelClassName="sm:max-w-xl">
        {selected && (
          <div>
            <div className="rounded-2xl border border-zinc-800 bg-zinc-950/55 p-4">
              <p className="text-sm leading-6 text-zinc-400">{selected.summary}</p>
              <p className="mt-3 text-xs font-medium text-zinc-600">Published release v{selected.current_release.version}</p>
            </div>

            <div className="mt-5">
              <Input label="Event name" value={title} onChange={(event) => setTitle(event.target.value)} maxLength={300} autoFocus />
              <p className="mt-2 text-xs leading-5 text-zinc-500">This names your Event. The template remains a starting point, not a permanent restriction.</p>
            </div>

            <div className="mt-5">
              <h3 className="text-sm font-semibold text-zinc-200">Your Event will begin with</h3>
              {detailLoading ? (
                <div className="mt-3 flex items-center gap-2 text-sm text-zinc-500"><LoaderCircle size={15} className="animate-spin" /> Loading setup details</div>
              ) : detail?.release.guided_jobs.length ? (
                <ul className="mt-3 space-y-2">
                  {detail.release.guided_jobs.map((job) => (
                    <li key={job.key} className="flex items-start gap-2 text-sm leading-5 text-zinc-400"><Check size={15} className="mt-0.5 shrink-0 text-accent-700" /><span>{job.title}</span></li>
                  ))}
                </ul>
              ) : (
                <p className="mt-2 text-sm leading-6 text-zinc-500">A focused starting Surface, Event context, and useful first actions from the reviewed release.</p>
              )}
            </div>

            {installError && <p className="mt-5 rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-400">{installError}</p>}

            <div className="mt-6 flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
              <Button variant="ghost" onClick={closeTemplate} disabled={installing}>Not now</Button>
              <Button onClick={() => void installSelected()} disabled={installing || !title.trim()}>
                {installing ? <LoaderCircle size={16} className="animate-spin" /> : <ArrowRight size={16} />}
                {installing ? 'Starting Event…' : 'Start this Event'}
              </Button>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
}
