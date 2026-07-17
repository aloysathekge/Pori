import { useEffect, useState, type ComponentType } from 'react';
import { NavLink, Outlet, useLocation, useNavigate } from 'react-router-dom';
import {
  Bot,
  CalendarClock,
  ChevronRight,
  FileText,
  Folder,
  LogOut,
  Menu,
  MessageSquarePlus,
  PanelLeftClose,
  Settings,
  X,
} from 'lucide-react';
import { listEvents, type EventSummary } from '@/api/events';
import { createConversation } from '@/api/conversations';
import { useAuth } from '@/contexts/useAuth';
import { Button } from '@/components/ui/Button';
import { ThemeToggle } from '@/components/ThemeToggle';
import { AloyMark, ChatIcon, EventIcon, MemoryIcon, TodayIcon } from '@/components/icons';

const utilityItems = [
  { to: '/files', icon: FileText, label: 'Files' },
  { to: '/memory', icon: MemoryIcon, label: 'Memory' },
  { to: '/schedules', icon: CalendarClock, label: 'Schedules' },
  { to: '/agents', icon: Bot, label: 'Agents' },
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

export function AppLayout() {
  const { signOut } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  const [events, setEvents] = useState<EventSummary[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarPeek, setSidebarPeek] = useState(false);
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

  function toggleExpanded() {
    setExpanded((value) => {
      localStorage.setItem('aloy.nav', value ? 'auto' : 'full');
      setSidebarPeek(value);
      return !value;
    });
  }

  const compact = false;
  const closeMobile = () => setSidebarOpen(false);
  const dedicatedEvents = events.filter((event) => !event.is_life);

  async function startConversation() {
    setActionError('');
    try {
      const conversation = await createConversation({});
      closeMobile();
      navigate(`/chat/${conversation.id}`);
    } catch (cause) {
      setActionError(cause instanceof Error ? cause.message : 'Could not start a conversation');
    }
  }

  function startEvent() {
    closeMobile();
    navigate('/events/new');
  }

  return (
    <div className="flex h-screen overflow-hidden bg-zinc-950 text-zinc-100">
      {sidebarOpen && (
        <button
          type="button"
          aria-label="Close navigation"
          className="fixed inset-0 z-30 bg-black/25 backdrop-blur-sm lg:hidden"
          onClick={closeMobile}
        />
      )}

      {!expanded && (
        <div
          className="fixed inset-y-0 left-0 z-30 hidden w-3 lg:block"
          onMouseEnter={() => setSidebarPeek(true)}
          aria-hidden="true"
        />
      )}

      <aside
        onMouseLeave={() => { if (!expanded) setSidebarPeek(false); }}
        className={`fixed inset-y-0 left-0 z-40 flex w-72 flex-col border-r border-zinc-800 bg-zinc-950 transition-transform duration-200 ${
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
              <RailLink
                key={event.id}
                to={`/events/${event.id}`}
                icon={Folder}
                label={event.title}
                compact={compact}
                onClick={closeMobile}
              />
            ))}
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

        <div className={`shrink-0 border-t border-zinc-800 p-2.5 ${compact ? 'space-y-1' : 'space-y-0.5'}`}>
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
        <header className="flex h-12 shrink-0 items-center border-b border-zinc-800 px-3 lg:hidden">
          <button
            type="button"
            className="rounded-md p-1.5 text-zinc-400 hover:bg-zinc-800"
            aria-label="Open menu"
            onClick={() => setSidebarOpen(true)}
          >
            <Menu size={20} />
          </button>
          <span className="ml-3 font-display font-semibold">Aloy</span>
        </header>
        <main className="min-h-0 flex-1 overflow-hidden">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
