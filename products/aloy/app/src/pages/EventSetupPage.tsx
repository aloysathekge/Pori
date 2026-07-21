import { useEffect, useRef, useState } from 'react';
import {
  ArrowLeft,
  ArrowRight,
  Link2,
  LoaderCircle,
  Paperclip,
  Plug,
  Trash2,
  X,
} from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import { listConnections, type ConnectionScope, type ProviderInfo } from '@/api/connections';
import {
  addEventDraftConnection,
  addEventDraftLink,
  createEventDraft,
  currentEventDraft,
  promoteEventDraft,
  removeEventDraftContext,
  updateEventDraft,
  uploadEventDraftFile,
  type EventSetupContextItem,
  type EventSetupDraft,
  type EventSetupMode,
} from '@/api/eventSetup';
import { AloyMark } from '@/components/icons';
import { FileTypeIcon } from '@/components/files/FileVisual';
import { Button } from '@/components/ui/Button';

const INPUT = 'w-full rounded-xl border border-zinc-700 bg-zinc-900 px-4 py-3 text-zinc-100 placeholder:text-zinc-500 focus:border-accent-600 focus:outline-none focus:ring-2 focus:ring-accent-600/15';

function sizeLabel(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function contextIcon(item: EventSetupContextItem) {
  if (item.kind === 'file') return <FileTypeIcon file={{ name: item.label, content_type: item.content_type ?? undefined }} size={16} />;
  if (item.kind === 'link') return <Link2 size={16} />;
  return <Plug size={16} />;
}

export function EventSetupPage() {
  const navigate = useNavigate();
  const [draft, setDraft] = useState<EventSetupDraft | null>(null);
  const [mode, setMode] = useState<EventSetupMode>('simple');
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [showLink, setShowLink] = useState(false);
  const [showConnections, setShowConnections] = useState(false);
  const [linkUrl, setLinkUrl] = useState('');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState('');
  const fileInput = useRef<HTMLInputElement>(null);
  const draftId = draft?.id;
  const draftStatus = draft?.status;

  useEffect(() => {
    let active = true;
    async function load() {
      try {
        const existing = await currentEventDraft();
        const value = existing || await createEventDraft();
        if (!active) return;
        setDraft(value);
        setMode(value.mode);
        setTitle(value.title);
        setDescription(value.description);
      } catch (cause) {
        if (active) setError(cause instanceof Error ? cause.message : 'Could not open Event setup');
      } finally {
        if (active) setLoading(false);
      }
    }
    void load();
    void listConnections().then((items) => {
      if (active) setProviders(items);
    }).catch(() => undefined);
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (!draftId || loading || draftStatus !== 'open') return;
    const timeout = window.setTimeout(() => {
      void updateEventDraft(draftId, { title, description, mode })
        .then(setDraft)
        .catch(() => undefined);
    }, 500);
    return () => window.clearTimeout(timeout);
  }, [description, draftId, draftStatus, loading, mode, title]);

  async function addFiles(files: FileList | File[]) {
    if (!draft || !files.length) return;
    setUploading(true);
    setError('');
    try {
      for (const file of Array.from(files)) {
        const item = await uploadEventDraftFile(draft.id, file, setUploadProgress);
        setDraft((current) => current ? {
          ...current,
          context_items: [...current.context_items, item],
        } : current);
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not attach that file');
    } finally {
      setUploading(false);
      setUploadProgress(0);
      if (fileInput.current) fileInput.current.value = '';
    }
  }

  async function addLink() {
    if (!draft || !linkUrl.trim()) return;
    setError('');
    try {
      const item = await addEventDraftLink(draft.id, linkUrl.trim());
      setDraft((current) => current ? {
        ...current,
        context_items: [...current.context_items, item],
      } : current);
      setLinkUrl('');
      setShowLink(false);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not add that link');
    }
  }

  async function attachConnection(
    provider: ProviderInfo,
    connectionScope: ConnectionScope,
  ) {
    if (!draft) return;
    setError('');
    try {
      const email = connectionScope === 'user' ? provider.account_email : provider.org_account_email;
      const item = await addEventDraftConnection(
        draft.id,
        provider.provider,
        connectionScope,
        email || provider.label,
      );
      setDraft((current) => current ? {
        ...current,
        context_items: current.context_items.some((value) => value.id === item.id)
          ? current.context_items
          : [...current.context_items, item],
      } : current);
      setShowConnections(false);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not attach that connection');
    }
  }

  async function removeItem(itemId: string) {
    if (!draft) return;
    try {
      await removeEventDraftContext(draft.id, itemId);
      setDraft((current) => current ? {
        ...current,
        context_items: current.context_items.filter((item) => item.id !== itemId),
      } : current);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not remove that context');
    }
  }

  async function submit() {
    if (!draft || !title.trim()) return;
    setSaving(true);
    setError('');
    try {
      await updateEventDraft(draft.id, { title: title.trim(), description, mode });
      const event = await promoteEventDraft(draft.id);
      navigate(`/events/${event.id}`, { replace: true });
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not create this Event');
      setSaving(false);
    }
  }

  if (loading) {
    return <div className="flex h-full items-center justify-center bg-zinc-950"><LoaderCircle className="animate-spin text-accent-600" /></div>;
  }

  const connected = providers.flatMap((provider) => [
    ...(provider.connected ? [{ provider, scope: 'user' as const }] : []),
    ...(provider.org_connected ? [{ provider, scope: 'org' as const }] : []),
  ]);

  return (
    <div className="h-full overflow-y-auto bg-zinc-950">
      <div className="mx-auto flex min-h-full max-w-5xl flex-col px-5 py-5 sm:px-8 lg:py-6">
        <header className="sticky top-0 z-20 -mx-5 flex items-center justify-between gap-3 border-b border-zinc-800/80 bg-zinc-950/95 px-5 py-2 backdrop-blur sm:-mx-8 sm:px-8 lg:static lg:mx-0 lg:border-0 lg:bg-transparent lg:px-0 lg:py-0">
          <button type="button" onClick={() => navigate('/today')} className="flex items-center gap-2 text-sm text-zinc-500 hover:text-zinc-200">
            <ArrowLeft size={16} /> <span className="hidden sm:inline">Back to </span>Today
          </button>
          <Button onClick={() => void submit()} disabled={saving || !draft || !title.trim()} className="px-5">
            {saving ? <LoaderCircle size={16} className="animate-spin" /> : <ArrowRight size={16} />}
            {saving ? 'Creating…' : 'Create Event'}
          </Button>
        </header>

        <main className="mx-auto mt-5 w-full max-w-3xl pb-6 sm:mt-7">
          <div className="flex items-center gap-2 text-accent-600"><AloyMark size={23} /><span className="text-sm font-semibold">Create an Event</span></div>
          <h1 className="mt-2 font-display text-2xl font-semibold tracking-tight text-zinc-100 sm:text-4xl">
            {mode === 'simple' ? 'What do you want to keep moving?' : 'Tell Aloy what you are taking on'}
          </h1>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-zinc-400">
            An Event is your dedicated, ongoing space where Aloy helps you manage something important over time.
          </p>

          <section className="mt-5 space-y-4">
            <div>
              <label htmlFor="event-name" className="mb-2 block text-sm font-semibold text-zinc-300">Event name</label>
              <span className="relative block">
                <input id="event-name" autoFocus value={title} onChange={(event) => setTitle(event.target.value)} className={`${INPUT} pr-32 text-lg`} placeholder="My university semester" maxLength={300} />
                <button type="button" onClick={() => setMode((value) => value === 'simple' ? 'assisted' : 'simple')} className="absolute inset-y-1.5 right-1.5 flex items-center gap-2 rounded-lg border-l border-zinc-700 px-3 text-sm font-medium text-zinc-500 transition hover:bg-zinc-800 hover:text-accent-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-600/30">
                  <AloyMark size={16} /> {mode === 'simple' ? 'Ask Aloy' : 'Start simple'}
                </button>
              </span>
            </div>

            <div>
              <div className="mb-2 flex flex-wrap items-end justify-between gap-2">
                <div><label htmlFor="event-context" className="block text-sm font-semibold text-zinc-300">Give Aloy some context</label><p className="mt-1 text-xs text-zinc-500">Optional. Add what Aloy should understand or use inside this Event.</p></div>
                <span className="text-xs text-zinc-600">Saved as you go</span>
              </div>
              <div
                className="overflow-hidden rounded-2xl border border-zinc-700 bg-zinc-900 focus-within:border-accent-600 focus-within:ring-2 focus-within:ring-accent-600/15"
                onDragOver={(event) => event.preventDefault()}
                onDrop={(event) => { event.preventDefault(); void addFiles(event.dataTransfer.files); }}
              >
                <textarea id="event-context" value={description} onChange={(event) => setDescription(event.target.value)} className="min-h-28 w-full resize-none bg-transparent px-4 py-3 text-base leading-6 text-zinc-100 placeholder:text-zinc-500 focus:outline-none sm:text-sm" placeholder={mode === 'assisted' ? 'I’m struggling to manage university this semester. I need help with my timetable, assignments and exams…' : 'What is this Event about? What matters, what is already decided, and where should Aloy begin?'} maxLength={50_000} />
                <div className="flex flex-wrap items-center gap-1 border-t border-zinc-800 px-2 py-2">
                  <button type="button" onClick={() => fileInput.current?.click()} className="flex items-center gap-2 rounded-lg px-3 py-2 text-xs font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100"><Paperclip size={15} />{uploading ? `Uploading ${uploadProgress}%` : 'Files'}</button>
                  <button type="button" onClick={() => { setShowLink((value) => !value); setShowConnections(false); }} className="flex items-center gap-2 rounded-lg px-3 py-2 text-xs font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100"><Link2 size={15} />Link</button>
                  <button type="button" onClick={() => { setShowConnections((value) => !value); setShowLink(false); }} className="flex items-center gap-2 rounded-lg px-3 py-2 text-xs font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100"><Plug size={15} />Connection</button>
                  <input ref={fileInput} type="file" multiple className="hidden" onChange={(event) => { if (event.target.files) void addFiles(event.target.files); }} />
                </div>
              </div>

              {showLink && (
                <div className="relative mt-2 flex flex-col gap-2 rounded-xl border border-zinc-800 bg-zinc-900 p-2 sm:flex-row">
                  <input autoFocus type="url" value={linkUrl} onChange={(event) => setLinkUrl(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter') void addLink(); }} className="min-h-11 min-w-0 flex-1 bg-transparent px-2 pr-12 text-base text-zinc-100 placeholder:text-zinc-500 focus:outline-none sm:pr-2 sm:text-sm" placeholder="https://…" />
                  <Button size="sm" onClick={() => void addLink()} disabled={!linkUrl.trim()}>Add link</Button>
                  <button type="button" aria-label="Close link input" onClick={() => setShowLink(false)} className="absolute right-2 top-2 flex h-11 w-11 items-center justify-center rounded-lg text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200 sm:static"><X size={16} /></button>
                </div>
              )}

              {showConnections && (
                <div className="mt-2 rounded-xl border border-zinc-800 bg-zinc-900 p-3">
                  {connected.length ? <div className="space-y-1">{connected.map(({ provider, scope }) => (
                    <button key={`${provider.provider}-${scope}`} type="button" onClick={() => void attachConnection(provider, scope)} className="flex w-full flex-col items-start gap-2 rounded-lg px-3 py-3 text-left hover:bg-zinc-800 sm:flex-row sm:items-center sm:justify-between">
                      <span><span className="block text-sm font-medium text-zinc-200">{provider.label}</span><span className="block text-xs text-zinc-500">{scope === 'user' ? provider.account_email : provider.org_account_email || 'Organization connection'}</span></span>
                      <span className="text-xs font-medium text-accent-700">Use in this Event</span>
                    </button>
                  ))}</div> : <p className="text-sm text-zinc-500">No accounts are connected yet. <Link to="/connections" className="font-medium text-accent-700 hover:underline">Set up a connection</Link></p>}
                </div>
              )}
            </div>

            {draft && draft.context_items.length > 0 && (
              <div className="grid gap-2 sm:grid-cols-2">
                {draft.context_items.map((item) => (
                  <div key={item.id} className="flex min-w-0 items-center gap-3 rounded-xl border border-zinc-800 bg-zinc-900 px-3 py-2.5">
                    <span className="text-zinc-500">{contextIcon(item)}</span>
                    <span className="min-w-0 flex-1"><span className="block truncate text-sm font-medium text-zinc-300">{item.label}</span><span className="block text-xs text-zinc-600">{item.kind === 'file' ? sizeLabel(item.size_bytes) : item.status === 'pending' ? 'Ready to process' : 'Available to this Event'}</span></span>
                    <button type="button" aria-label={`Remove ${item.label}`} onClick={() => void removeItem(item.id)} className="rounded-lg p-2 text-zinc-600 hover:bg-zinc-800 hover:text-red-500"><Trash2 size={14} /></button>
                  </div>
                ))}
              </div>
            )}
          </section>

          {error && <p className="mt-4 rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-500">{error}</p>}
        </main>
      </div>
    </div>
  );
}
