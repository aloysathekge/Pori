import { useEffect, useState, type ComponentType } from 'react';
import { NavLink, Outlet, useLocation, useNavigate } from 'react-router-dom';
import {
  Bot,
  CalendarClock,
  ChevronRight,
  FileText,
  Folder,
  FolderPlus,
  LogOut,
  Menu,
  MessageSquare,
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
import { AloyMark, MemoryIcon, TodayIcon } from '@/components/icons';

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
  const [actionError, setActionError] = useState('');
  const [expanded, setExpanded] = useState(
    () => localStorage.getItem('aloy.nav') !== 'slim',
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
      localStorage.setItem('aloy.nav', value ? 'slim' : 'full');
      return !value;
    });
  }

  const compact = !expanded;
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

      <aside
        className={`fixed inset-y-0 left-0 z-40 flex w-72 flex-col border-r border-zinc-800 bg-zinc-950 transition-[width,transform] duration-200 lg:static lg:translate-x-0 ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        } ${expanded ? 'lg:w-72' : 'lg:w-[68px]'}`}
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
          <RailLink to="/chat" icon={MessageSquare} label="Chat" compact={compact} onClick={closeMobile} />
          <RailLink to="/today" icon={TodayIcon} label="Today" compact={compact} onClick={closeMobile} />

          <div className={`mt-3 grid gap-1 ${compact ? '' : 'grid-cols-2'}`}>
            <Button
              variant="ghost"
              size={compact ? 'icon' : undefined}
              className={compact ? 'w-full' : 'justify-start px-2 text-xs'}
              onClick={startConversation}
              title="New conversation"
            >
              <MessageSquarePlus size={16} />
              {!compact && <span>New chat</span>}
            </Button>
            <Button
              variant="ghost"
              size={compact ? 'icon' : undefined}
              className={compact ? 'w-full' : 'justify-start px-2 text-xs'}
              onClick={() => {
                closeMobile();
                navigate('/today?new=event');
              }}
              title="New event"
            >
              <FolderPlus size={16} />
              {!compact && <span>New event</span>}
            </Button>
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
            title={expanded ? 'Collapse sidebar' : 'Expand sidebar'}
          >
            {expanded ? <PanelLeftClose size={17} /> : <ChevronRight size={17} />}
            {!compact && <span>Collapse</span>}
          </button>
          <ThemeToggle expanded={expanded} />
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
