import { NavLink, Outlet } from 'react-router-dom';
import { FolderOpen, LogOut, Menu, PanelLeftClose, PanelLeftOpen, Plug, X } from 'lucide-react';
import { useState } from 'react';
import { useAuth } from '@/contexts/useAuth';
import { Button } from '@/components/ui/Button';
import { ThemeToggle } from '@/components/ThemeToggle';
import {
  AgentsIcon,
  AloyMark,
  ChatIcon,
  MemoryIcon,
  SchedulesIcon,
  SettingsIcon,
  SkillsIcon,
  TeamsIcon,
  TodayIcon,
  TracesIcon,
  UsageIcon,
} from '@/components/icons';

const navItems = [
  { to: '/today', icon: TodayIcon, label: 'Today' },
  { to: '/chat', icon: ChatIcon, label: 'Chat' },
  { to: '/files', icon: FolderOpen, label: 'My Files' },
  { to: '/agents', icon: AgentsIcon, label: 'Agents' },
  { to: '/skills', icon: SkillsIcon, label: 'Skills' },
  { to: '/schedules', icon: SchedulesIcon, label: 'Schedules' },
  { to: '/connections', icon: Plug, label: 'Connections' },
  { to: '/teams', icon: TeamsIcon, label: 'Teams' },
  { to: '/memory', icon: MemoryIcon, label: 'Memory' },
  { to: '/usage', icon: UsageIcon, label: 'Usage' },
  { to: '/traces', icon: TracesIcon, label: 'Traces' },
  { to: '/settings', icon: SettingsIcon, label: 'Settings' },
];

export function AppLayout() {
  const { signOut } = useAuth();
  const [sidebarOpen, setSidebarOpen] = useState(false); // mobile drawer
  // Desktop: full rail (labels) vs slim icon rail — the icon rail leaves the
  // content essentially screen-centered. Preference persists.
  const [expanded, setExpanded] = useState<boolean>(
    () => localStorage.getItem('aloy.nav') !== 'slim',
  );

  function toggleExpanded() {
    setExpanded((prev) => {
      localStorage.setItem('aloy.nav', prev ? 'slim' : 'full');
      return !prev;
    });
  }

  return (
    <div className="flex h-screen bg-zinc-950 text-zinc-100">
      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/30 backdrop-blur-sm lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar: full drawer on mobile; full-or-icon rail on desktop */}
      <aside
        className={`fixed inset-y-0 left-0 z-40 flex w-64 flex-col border-r border-zinc-800/80 bg-zinc-900/60 backdrop-blur transition-all duration-300 lg:static lg:translate-x-0 ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        } ${expanded ? 'lg:w-64' : 'lg:w-16'}`}
      >
        <div
          className={`flex h-16 items-center gap-3 border-b border-zinc-800/80 ${
            expanded ? 'px-5' : 'px-5 lg:justify-center lg:px-0'
          }`}
        >
          <AloyMark size={30} />
          <span
            className={`font-display text-xl font-semibold tracking-tight ${
              expanded ? '' : 'lg:hidden'
            }`}
          >
            Aloy
          </span>
          <button
            className="ml-auto rounded-lg p-1.5 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100 lg:hidden"
            aria-label="Close menu"
            onClick={() => setSidebarOpen(false)}
          >
            <X size={20} />
          </button>
        </div>

        <nav
          className={`flex-1 space-y-0.5 overflow-y-auto py-4 ${
            expanded ? 'px-3' : 'px-3 lg:px-2.5'
          }`}
        >
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              onClick={() => setSidebarOpen(false)}
              title={expanded ? undefined : item.label}
              className={({ isActive }) =>
                `group relative flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition-all duration-150 ${
                  expanded ? '' : 'lg:justify-center lg:px-0'
                } ${
                  isActive
                    ? 'bg-accent-600/15 text-accent-700'
                    : 'text-zinc-400 hover:bg-zinc-800/70 hover:text-zinc-100'
                }`
              }
            >
              {({ isActive }) => (
                <>
                  {isActive && (
                    <span className="absolute left-0 top-1/2 h-5 w-0.5 -translate-y-1/2 rounded-full bg-accent-500" />
                  )}
                  <item.icon
                    size={19}
                    className={
                      isActive
                        ? 'text-accent-600'
                        : 'text-zinc-500 transition-colors group-hover:text-zinc-300'
                    }
                  />
                  <span className={expanded ? '' : 'lg:hidden'}>
                    {item.label}
                  </span>
                </>
              )}
            </NavLink>
          ))}
        </nav>

        <div
          className={`space-y-1 border-t border-zinc-800/80 p-3 ${
            expanded ? '' : 'lg:p-2.5'
          }`}
        >
          {/* Desktop rail toggle */}
          <button
            type="button"
            onClick={toggleExpanded}
            title={expanded ? 'Collapse sidebar' : 'Expand sidebar'}
            className={`hidden w-full items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium text-zinc-400 transition-colors hover:bg-zinc-800/70 hover:text-zinc-100 lg:flex ${
              expanded ? '' : 'lg:justify-center lg:px-0'
            }`}
          >
            {expanded ? (
              <>
                <PanelLeftClose size={18} className="text-zinc-500" />
                Collapse
              </>
            ) : (
              <PanelLeftOpen size={18} className="text-zinc-500" />
            )}
          </button>
          <ThemeToggle expanded={expanded} />
          <Button
            variant="ghost"
            size="md"
            className={`w-full gap-3 text-zinc-400 ${
              expanded ? 'justify-start' : 'justify-start lg:justify-center'
            }`}
            onClick={signOut}
            title={expanded ? undefined : 'Sign Out'}
          >
            <LogOut size={18} />
            <span className={expanded ? '' : 'lg:hidden'}>Sign Out</span>
          </Button>
        </div>
      </aside>

      {/* Main content */}
      <div className="flex flex-1 flex-col overflow-hidden">
        <header className="flex h-14 items-center border-b border-zinc-800/80 px-4 lg:hidden">
          <button
            className="rounded-lg p-1.5 text-zinc-300 hover:bg-zinc-800"
            aria-label="Open menu"
            onClick={() => setSidebarOpen(true)}
          >
            <Menu size={22} />
          </button>
          <span className="ml-3 flex items-center gap-2">
            <AloyMark size={22} />
            <span className="font-display font-semibold">Aloy</span>
          </span>
        </header>
        <main className="flex-1 overflow-y-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
