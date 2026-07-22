import { useEffect, useState, type ComponentType } from 'react';
import { NavLink, Outlet, useLocation, useNavigate } from 'react-router-dom';
import {
  Archive,
  CalendarClock,
  ChevronRight,
  FileText,
  LogOut,
  Menu,
  MessageSquarePlus,
  MoreHorizontal,
  PanelLeftClose,
  Plus,
  Plug,
  Settings,
  X,
} from 'lucide-react';
import { listEvents, updateEvent, type EventSummary } from '@/api/events';
import { createConversation } from '@/api/conversations';
import { useAuth } from '@/contexts/useAuth';
import { useToast } from '@/contexts/toast';
import { WorkspaceFocusContext } from '@/contexts/WorkspaceFocusContext';
import { Button } from '@/components/ui/Button';
import { Modal } from '@/components/ui/Modal';
import { Spinner } from '@/components/ui/Spinner';
import { EventCover } from '@/components/events/EventCover';
import { ThemeToggle } from '@/components/ThemeToggle';
import { AloyMark, ChatIcon, EventIcon, MemoryIcon, TodayIcon } from '@/components/icons';

const utilityItems = [
  { to: '/files', icon: FileText, label: 'Files' },
  { to: '/memory', icon: MemoryIcon, label: 'Memory' },
  { to: '/schedules', icon: CalendarClock, label: 'Schedules' },
  { to: '/connections', icon: Plug, label: 'Connections' },
  { to: '/settings', icon: Settings, label: 'Settings' },
];

function RailLink({
  to,
  icon: Icon,
  label,
  compact = false,
  onClick,
}: {
  to: string;
  icon: ComponentType<{ size?: number; className?: string }>;
  label: string;
  compact?: boolean;
  onClick?: () => void;
}) {
  return (
    <NavLink
      to={to}
      onClick={onClick}
      title={compact ? label : undefined}
      className={({ isActive }) =>
        `group flex min-h-9 items-center gap-2.5 rounded-lg px-2.5 text-sm transition-colors ${
          compact ? 'justify-center' : ''
        } ${
          isActive
            ? 'bg-zinc-800 text-zinc-100'
            : 'text-zinc-400 hover:bg-zinc-800/70 hover:text-zinc-200'
        }`
      }
    >
      <Icon size={17} className="shrink-0" />
      {!compact && <span className="truncate">{label}</span>}
    </NavLink>
  );
}

function EventRailLink({
  event,
  duplicateTitle,
  menuOpen,
  onNavigate,
  onToggleMenu,
  onOpenSettings,
  onArchive,
}: {
  event: EventSummary;
  duplicateTitle: boolean;
  menuOpen: boolean;
  onNavigate: () => void;
  onToggleMenu: () => void;
  onOpenSettings: () => void;
  onArchive: () => void;
}) {
  return (
    <div className="group/event relative">
      <NavLink
        to={`/events/${event.id}`}
        onClick={onNavigate}
        title={duplicateTitle ? `${event.title} · Created ${new Date(event.created_at).toLocaleDateString()}` : event.title}
        className={({ isActive }) => `flex min-h-11 items-center gap-2.5 rounded-lg px-2 pr-10 text-sm transition-colors ${isActive ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-400 hover:bg-zinc-800/70 hover:text-zinc-200'}`}
      >
        <EventCover event={event} className="h-7 w-9 shrink-0 rounded-md border border-zinc-800" />
        <span className="min-w-0 flex-1">
          <span className="block truncate">{event.title}</span>
          {duplicateTitle && (
            <span className="block truncate text-[10px] text-zinc-600">
              Created {new Date(event.created_at).toLocaleDateString()}
            </span>
          )}
        </span>
      </NavLink>
      <button
        type="button"
        onClick={onToggleMenu}
        className={`absolute right-1 top-1 flex h-9 w-9 items-center justify-center rounded-lg text-zinc-500 transition-colors hover:bg-zinc-700 hover:text-zinc-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-500 ${menuOpen ? 'bg-zinc-700 text-zinc-100' : 'opacity-70 group-hover/event:opacity-100'}`}
        aria-label={`Actions for ${event.title}`}
        aria-expanded={menuOpen}
        title={`Manage ${event.title}`}
      >
        <MoreHorizontal size={17} />
      </button>
      {menuOpen && (
        <div className="relative z-20 mx-1 mb-1 mt-1 space-y-1 rounded-xl border border-zinc-700 bg-zinc-900 p-1.5 shadow-xl" role="menu" aria-label={`${event.title} actions`}>
          <button type="button" role="menuitem" onClick={onOpenSettings} className="flex min-h-10 w-full items-center gap-2.5 rounded-lg px-3 text-left text-sm text-zinc-300 hover:bg-zinc-800 hover:text-white">
            <Settings size={15} /> Event settings
          </button>
          <button type="button" role="menuitem" onClick={onArchive} className="flex min-h-10 w-full items-center gap-2.5 rounded-lg px-3 text-left text-sm text-red-400 hover:bg-red-500/10 hover:text-red-300">
            <Archive size={15} /> Archive Event…
          </button>
        </div>
      )}
    </div>
  );
}

export function AppLayout() {
  const { signOut } = useAuth();
  const { showToast } = useToast();
  const location = useLocation();
  const navigate = useNavigate();
  const [events, setEvents] = useState<EventSummary[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarPeek, setSidebarPeek] = useState(false);
  const [workspaceFocused, setWorkspaceFocused] = useState(false);
  const [mobileSheet, setMobileSheet] = useState<'create' | 'events' | null>(null);
  const [eventMenuId, setEventMenuId] = useState<string | null>(null);
  const [archiveTarget, setArchiveTarget] = useState<EventSummary | null>(null);
  const [archiving, setArchiving] = useState(false);
  const [archiveError, setArchiveError] = useState('');
  const [actionError, setActionError] = useState('');
  const [expanded, setExpanded] = useState(
    () => !['slim', 'auto'].includes(localStorage.getItem('aloy.nav') || ''),
  );

  useEffect(() => {
    let cancelled = false;
    listEvents()
      .then((items) => {
        if (!cancelled) setEvents(items);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [location.pathname]);

  useEffect(() => {
    if (!mobileSheet) return;
    const previousOverflow = document.body.style.overflow;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setMobileSheet(null);
    };
    document.body.style.overflow = 'hidden';
    window.addEventListener('keydown', closeOnEscape);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener('keydown', closeOnEscape);
    };
  }, [mobileSheet]);

  function toggleExpanded() {
    setExpanded((value) => {
      localStorage.setItem('aloy.nav', value ? 'auto' : 'full');
      setSidebarPeek(value);
      return !value;
    });
  }

  const compact = false;
  const closeMobile = () => {
    setSidebarOpen(false);
    setEventMenuId(null);
  };
  const dedicatedEvents = events.filter((event) => !event.is_life);
  const eventTitleCounts = dedicatedEvents.reduce((counts, event) => {
    counts.set(event.title, (counts.get(event.title) ?? 0) + 1);
    return counts;
  }, new Map<string, number>());
  const activeEvent = dedicatedEvents.find((event) => location.pathname === `/events/${event.id}`);
  const mobileTitle = activeEvent?.title
    || (location.pathname.startsWith('/events/new') || location.pathname.startsWith('/events/start') ? 'New Event'
      : location.pathname.startsWith('/events/archived') ? 'Archived Events'
      : location.pathname.startsWith('/chat') ? 'Life'
        : location.pathname.startsWith('/today') ? 'Today'
          : utilityItems.find((item) => location.pathname.startsWith(item.to))?.label
            || 'Aloy');

  async function startConversation() {
    setActionError('');
    try {
      const conversation = await createConversation({});
      closeMobile();
      setMobileSheet(null);
      navigate(`/chat/${conversation.id}`);
    } catch (cause) {
      setActionError(cause instanceof Error ? cause.message : 'Could not start a conversation');
    }
  }

  function startEvent() {
    closeMobile();
    setMobileSheet(null);
    navigate('/events/start');
  }

  function openEventSettings(event: EventSummary) {
    setEventMenuId(null);
    setSidebarOpen(false);
    setMobileSheet(null);
    navigate(`/events/${event.id}?panel=settings`);
  }

  function requestEventArchive(event: EventSummary) {
    setEventMenuId(null);
    setArchiveError('');
    setArchiveTarget(event);
  }

  async function archiveEvent() {
    if (!archiveTarget) return;
    setArchiving(true);
    setArchiveError('');
    try {
      await updateEvent(archiveTarget.id, { lifecycle: 'archived' });
      setEvents((current) => current.filter((event) => event.id !== archiveTarget.id));
      showToast({
        tone: 'success',
        title: `${archiveTarget.title} archived`,
        description: 'Its work is paused. Restore or permanently delete it from Archived Events.',
      });
      if (location.pathname === `/events/${archiveTarget.id}`) {
        navigate('/today', { replace: true });
      }
      setArchiveTarget(null);
      setSidebarOpen(false);
    } catch (cause) {
      setArchiveError(cause instanceof Error ? cause.message : 'The Event could not be archived.');
    } finally {
      setArchiving(false);
    }
  }

  return (
    <WorkspaceFocusContext.Provider value={{ focused: workspaceFocused, setFocused: setWorkspaceFocused }}>
    <div className="flex h-[100dvh] overflow-hidden bg-zinc-950 text-zinc-100">
      {!workspaceFocused && sidebarOpen && (
        <button
          type="button"
          aria-label="Close navigation"
          className="fixed inset-0 z-30 bg-black/25 backdrop-blur-sm lg:hidden"
          onClick={closeMobile}
        />
      )}

      {!workspaceFocused && !expanded && (
        <div
          className="fixed inset-y-0 left-0 z-30 hidden w-3 lg:block"
          onMouseEnter={() => setSidebarPeek(true)}
          aria-hidden="true"
        />
      )}

      <aside
        onMouseLeave={() => { if (!expanded) setSidebarPeek(false); }}
        className={`fixed inset-y-0 left-0 z-40 ${workspaceFocused ? 'hidden' : 'flex'} w-[min(20rem,calc(100vw-1.5rem))] flex-col border-r border-zinc-800 bg-zinc-950 pt-[env(safe-area-inset-top)] transition-transform duration-200 ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        } ${expanded ? 'lg:static lg:translate-x-0' : sidebarPeek ? 'lg:fixed lg:translate-x-0 lg:shadow-2xl' : 'lg:fixed lg:-translate-x-full'}`}
      >
        <div className={`flex h-14 shrink-0 items-center border-b border-zinc-800 ${compact ? 'justify-center px-2' : 'px-4'}`}>
          <AloyMark size={25} />
          {!compact && <span className="ml-2.5 font-display text-lg font-semibold">Aloy</span>}
          <button
            type="button"
            onClick={closeMobile}
            aria-label="Close menu"
            className="ml-auto rounded-md p-1.5 text-zinc-500 hover:bg-zinc-800 lg:hidden"
          >
            <X size={18} />
          </button>
        </div>

        <div className={`flex-1 overflow-y-auto ${compact ? 'px-2' : 'px-3'} py-3`}>
          {compact ? (
            <RailLink to="/chat" icon={ChatIcon} label="Life" compact onClick={closeMobile} />
          ) : (
            <div className="group relative">
              <NavLink
                to="/chat"
                onClick={closeMobile}
                className={({ isActive }) =>
                  `flex min-h-9 items-center gap-2.5 rounded-lg px-2.5 pr-10 text-sm transition-colors ${
                    isActive
                      ? 'bg-zinc-800 text-zinc-100'
                      : 'text-zinc-400 hover:bg-zinc-800/70 hover:text-zinc-200'
                  }`
                }
              >
                <ChatIcon size={17} className="shrink-0" />
                <span>Life</span>
              </NavLink>
              <button
                type="button"
                onClick={startConversation}
                aria-label="Start a new conversation"
                title="New conversation"
                className="absolute right-1 top-1/2 flex h-7 w-7 -translate-y-1/2 items-center justify-center rounded-md text-zinc-500 opacity-70 transition-colors hover:bg-zinc-700 hover:text-zinc-200 group-hover:opacity-100 focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-500"
              >
                <MessageSquarePlus size={15} />
              </button>
            </div>
          )}
          <RailLink to="/today" icon={TodayIcon} label="Today" compact={compact} onClick={closeMobile} />

          <div className="mt-3">
            <button
              type="button"
              onClick={startEvent}
              title={compact ? 'Start a new Event workspace' : undefined}
              className={`group w-full border border-zinc-800 bg-zinc-900 text-left transition-colors hover:border-accent-500/45 hover:bg-zinc-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-500 ${
                compact
                  ? 'flex h-10 items-center justify-center rounded-xl text-accent-400'
                  : 'flex h-11 items-center gap-2.5 rounded-xl px-2.5'
              }`}
            >
              {compact ? (
                <EventIcon size={18} />
              ) : (
                <>
                  <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-accent-500/10 text-accent-300 transition-colors group-hover:bg-accent-500/15">
                    <EventIcon size={17} />
                  </span>
                  <span className="text-sm font-semibold text-zinc-200">New Event</span>
                  <span className="ml-auto text-[9px] font-semibold uppercase tracking-[0.12em] text-zinc-600 group-hover:text-zinc-500">
                    Workspace
                  </span>
                </>
              )}
            </button>
          </div>
          {!compact && actionError && (
            <p className="mt-2 px-2 text-xs text-red-400">{actionError}</p>
          )}

          {!compact && (
            <p className="mb-1 mt-5 px-2.5 text-[11px] font-semibold uppercase tracking-[0.12em] text-zinc-500">
              Events
            </p>
          )}
          <div className={compact ? 'mt-3 space-y-1' : 'space-y-0.5'}>
            {dedicatedEvents.map((event) => (
              <EventRailLink
                key={event.id}
                event={event}
                duplicateTitle={(eventTitleCounts.get(event.title) ?? 0) > 1}
                menuOpen={eventMenuId === event.id}
                onNavigate={closeMobile}
                onToggleMenu={() => setEventMenuId((current) => current === event.id ? null : event.id)}
                onOpenSettings={() => openEventSettings(event)}
                onArchive={() => requestEventArchive(event)}
              />
            ))}
            <RailLink
              to="/events/archived"
              icon={Archive}
              label="Archived"
              compact={compact}
              onClick={closeMobile}
            />
          </div>

          {!compact && (
            <p className="mb-1 mt-5 px-2.5 text-[11px] font-semibold uppercase tracking-[0.12em] text-zinc-500">
              Workspace
            </p>
          )}
          <div className={compact ? 'mt-3 space-y-1 border-t border-zinc-800 pt-3' : 'space-y-0.5'}>
            {utilityItems.map((item) => (
              <RailLink key={item.to} {...item} compact={compact} onClick={closeMobile} />
            ))}
          </div>
        </div>

        <div className={`shrink-0 border-t border-zinc-800 p-2.5 pb-[max(0.625rem,env(safe-area-inset-bottom))] ${compact ? 'space-y-1' : 'space-y-0.5'}`}>
          <button
            type="button"
            onClick={toggleExpanded}
            className={`hidden min-h-9 w-full items-center gap-2.5 rounded-lg px-2.5 text-sm text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 lg:flex ${compact ? 'justify-center' : ''}`}
            title={expanded ? 'Auto-hide sidebar' : 'Keep sidebar open'}
          >
            {expanded ? <PanelLeftClose size={17} /> : <ChevronRight size={17} />}
            {!compact && <span>{expanded ? 'Auto-hide' : 'Keep open'}</span>}
          </button>
          <ThemeToggle expanded />
          <Button
            variant="ghost"
            className={`w-full gap-2.5 text-zinc-400 ${compact ? 'justify-center' : 'justify-start'}`}
            onClick={signOut}
            title={compact ? 'Sign out' : undefined}
          >
            <LogOut size={17} />
            {!compact && <span>Sign out</span>}
          </Button>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <header className={`${workspaceFocused ? 'hidden' : 'flex'} min-h-12 shrink-0 items-center border-b border-zinc-800 px-2 pt-[env(safe-area-inset-top)] lg:hidden`}>
          <button
            type="button"
            className="flex h-11 w-11 items-center justify-center rounded-lg text-zinc-400 hover:bg-zinc-800"
            aria-label="Open menu"
            onClick={() => setSidebarOpen(true)}
          >
            <Menu size={20} />
          </button>
          <AloyMark size={22} className="ml-1 shrink-0" />
          <span className="ml-2 min-w-0 truncate font-display font-semibold">{mobileTitle}</span>
        </header>
        <main className="min-h-0 flex-1 overflow-hidden">
          <Outlet />
        </main>

        <nav
          aria-label="Primary mobile navigation"
          className={`${workspaceFocused ? 'hidden' : 'grid'} shrink-0 grid-cols-5 border-t border-zinc-800 bg-zinc-950/95 pb-[env(safe-area-inset-bottom)] backdrop-blur lg:hidden`}
        >
          <NavLink to="/today" className={({ isActive }) => `flex min-h-14 flex-col items-center justify-center gap-1 text-[10px] font-medium ${isActive ? 'text-accent-700' : 'text-zinc-500'}`}>
            <TodayIcon size={20} /><span>Today</span>
          </NavLink>
          <NavLink to="/chat" className={({ isActive }) => `flex min-h-14 flex-col items-center justify-center gap-1 text-[10px] font-medium ${isActive ? 'text-accent-700' : 'text-zinc-500'}`}>
            <ChatIcon size={20} /><span>Life</span>
          </NavLink>
          <button type="button" onClick={() => setMobileSheet('create')} className="flex min-h-14 flex-col items-center justify-center gap-1 text-[10px] font-semibold text-zinc-300" aria-label="Create new" aria-expanded={mobileSheet === 'create'}>
            <span className="flex h-8 w-8 items-center justify-center rounded-full bg-accent-600 text-white shadow-sm"><Plus size={18} /></span>
            <span>New</span>
          </button>
          <button type="button" onClick={() => setMobileSheet('events')} className={`flex min-h-14 flex-col items-center justify-center gap-1 text-[10px] font-medium ${activeEvent ? 'text-accent-700' : 'text-zinc-500'}`} aria-label="Open Events" aria-expanded={mobileSheet === 'events'}>
            <EventIcon size={20} /><span>Events</span>
          </button>
          <button type="button" onClick={() => setSidebarOpen(true)} className="flex min-h-14 flex-col items-center justify-center gap-1 text-[10px] font-medium text-zinc-500" aria-label="More navigation">
            <MoreHorizontal size={21} /><span>More</span>
          </button>
        </nav>
      </div>

      {!workspaceFocused && mobileSheet && (
        <div className="fixed inset-0 z-50 flex items-end lg:hidden">
          <button type="button" className="absolute inset-0 bg-black/35 backdrop-blur-[1px]" onClick={() => setMobileSheet(null)} aria-label="Close menu" />
          <section className="relative z-10 max-h-[min(78dvh,42rem)] w-full overflow-y-auto rounded-t-3xl border-x border-t border-zinc-800 bg-zinc-900 px-4 pb-[max(1rem,env(safe-area-inset-bottom))] pt-3 shadow-2xl" role="dialog" aria-modal="true" aria-label={mobileSheet === 'create' ? 'Create' : 'Events'}>
            <div className="mx-auto mb-4 h-1 w-10 rounded-full bg-zinc-700" />
            <div className="mb-3 flex items-center justify-between">
              <div>
                <h2 className="font-display text-lg font-semibold text-zinc-100">{mobileSheet === 'create' ? 'Start something' : 'Your Events'}</h2>
                <p className="mt-0.5 text-xs text-zinc-500">{mobileSheet === 'create' ? 'Choose a conversation or a durable workspace.' : 'Open an ongoing area of your life.'}</p>
              </div>
              <button type="button" onClick={() => setMobileSheet(null)} className="flex h-11 w-11 items-center justify-center rounded-xl text-zinc-500 hover:bg-zinc-800" aria-label="Close"><X size={18} /></button>
            </div>
            {mobileSheet === 'create' ? (
              <div className="space-y-2">
                {actionError && <p className="rounded-xl bg-red-500/10 px-3 py-2 text-xs text-red-400">{actionError}</p>}
                <button type="button" onClick={() => void startConversation()} className="flex min-h-16 w-full items-center gap-3 rounded-2xl border border-zinc-800 bg-zinc-950/50 px-4 text-left">
                  <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-zinc-800 text-zinc-300"><MessageSquarePlus size={19} /></span>
                  <span><span className="block text-sm font-semibold text-zinc-100">New conversation</span><span className="mt-0.5 block text-xs text-zinc-500">Talk with Aloy in Life</span></span>
                </button>
                <button type="button" onClick={startEvent} className="flex min-h-16 w-full items-center gap-3 rounded-2xl border border-accent-600/25 bg-accent-600/5 px-4 text-left">
                  <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-accent-600/10 text-accent-700"><EventIcon size={19} /></span>
                  <span><span className="block text-sm font-semibold text-zinc-100">New Event</span><span className="mt-0.5 block text-xs text-zinc-500">Create a dedicated ongoing workspace</span></span>
                </button>
              </div>
            ) : dedicatedEvents.length ? (
              <div className="space-y-1">
                {dedicatedEvents.map((event) => (
                  <button key={event.id} type="button" onClick={() => { setMobileSheet(null); navigate(`/events/${event.id}`); }} className="flex min-h-16 w-full items-center gap-3 rounded-2xl px-2 text-left hover:bg-zinc-800/70">
                    <EventCover event={event} className="h-11 w-14 shrink-0 rounded-xl border border-zinc-800" />
                    <span className="min-w-0 flex-1"><span className="block truncate text-sm font-semibold text-zinc-100">{event.title}</span><span className="mt-0.5 block truncate text-xs text-zinc-500">{(eventTitleCounts.get(event.title) ?? 0) > 1 ? `Created ${new Date(event.created_at).toLocaleDateString()}` : event.summary || event.phase || 'Active Event'}</span></span>
                  </button>
                ))}
                <button type="button" onClick={() => { setMobileSheet(null); navigate('/events/archived'); }} className="flex min-h-14 w-full items-center gap-3 rounded-2xl px-2 text-left text-zinc-500 hover:bg-zinc-800/70 hover:text-zinc-200">
                  <span className="flex h-11 w-14 shrink-0 items-center justify-center rounded-xl border border-zinc-800 bg-zinc-950/60"><Archive size={18} /></span>
                  <span className="text-sm font-semibold">Archived Events</span>
                </button>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="rounded-2xl border border-dashed border-zinc-800 px-4 py-8 text-center"><p className="text-sm text-zinc-400">No dedicated Events yet.</p><button type="button" onClick={startEvent} className="mt-3 text-sm font-semibold text-accent-700">Create your first Event</button></div>
                <button type="button" onClick={() => { setMobileSheet(null); navigate('/events/archived'); }} className="flex min-h-14 w-full items-center justify-center gap-2 rounded-2xl text-sm font-semibold text-zinc-500 hover:bg-zinc-800/70 hover:text-zinc-200"><Archive size={17} /> Archived Events</button>
              </div>
            )}
          </section>
        </div>
      )}

      <Modal
        open={archiveTarget !== null}
        onClose={() => { if (!archiving) setArchiveTarget(null); }}
        title={archiveTarget ? `Archive ${archiveTarget.title}?` : 'Archive Event?'}
      >
        <p className="text-sm leading-6 text-zinc-400">
          Aloy will stop waking for this Event, and it will disappear from Today and your active Events. Its conversation, tasks, files, memory, Trail, and Surface remain intact.
        </p>
        <p className="mt-3 text-xs leading-5 text-zinc-500">
          You can restore it—or permanently delete it—from Archived Events.
        </p>
        {archiveError && (
          <p role="alert" className="mt-3 rounded-lg border border-red-500/25 bg-red-500/10 px-3 py-2 text-xs text-red-400">
            {archiveError}
          </p>
        )}
        <div className="mt-5 flex justify-end gap-2">
          <Button variant="ghost" onClick={() => setArchiveTarget(null)} disabled={archiving}>Cancel</Button>
          <Button variant="danger" onClick={() => void archiveEvent()} disabled={archiving}>
            {archiving ? <Spinner className="h-4 w-4" /> : <Archive size={16} />}
            Archive Event
          </Button>
        </div>
      </Modal>
    </div>
    </WorkspaceFocusContext.Provider>
  );
}
