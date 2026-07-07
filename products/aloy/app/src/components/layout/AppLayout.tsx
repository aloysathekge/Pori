import { NavLink, Outlet } from 'react-router-dom';
import { LogOut, Menu, X } from 'lucide-react';
import { useState } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { Button } from '@/components/ui/Button';
import {
  AgentsIcon,
  AloyMark,
  ChatIcon,
  MemoryIcon,
  SchedulesIcon,
  SettingsIcon,
  SkillsIcon,
  TeamsIcon,
  TracesIcon,
  UsageIcon,
} from '@/components/icons';

const navItems = [
  { to: '/chat', icon: ChatIcon, label: 'Chat' },
  { to: '/agents', icon: AgentsIcon, label: 'Agents' },
  { to: '/skills', icon: SkillsIcon, label: 'Skills' },
  { to: '/schedules', icon: SchedulesIcon, label: 'Schedules' },
  { to: '/teams', icon: TeamsIcon, label: 'Teams' },
  { to: '/memory', icon: MemoryIcon, label: 'Memory' },
  { to: '/usage', icon: UsageIcon, label: 'Usage' },
  { to: '/traces', icon: TracesIcon, label: 'Traces' },
  { to: '/settings', icon: SettingsIcon, label: 'Settings' },
];

export function AppLayout() {
  const { signOut } = useAuth();
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="flex h-screen bg-zinc-950 text-zinc-100">
      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/60 backdrop-blur-sm lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`fixed inset-y-0 left-0 z-40 flex w-64 flex-col border-r border-zinc-800/80 bg-zinc-900/60 backdrop-blur transition-transform lg:static lg:translate-x-0 ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="flex h-16 items-center gap-3 border-b border-zinc-800/80 px-5">
          <AloyMark size={30} />
          <span className="font-display text-xl font-semibold tracking-tight">
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

        <nav className="flex-1 space-y-0.5 overflow-y-auto px-3 py-4">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              onClick={() => setSidebarOpen(false)}
              className={({ isActive }) =>
                `group relative flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition-all duration-150 ${
                  isActive
                    ? 'bg-accent-600/15 text-accent-300'
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
                        ? 'text-accent-400'
                        : 'text-zinc-500 transition-colors group-hover:text-zinc-300'
                    }
                  />
                  {item.label}
                </>
              )}
            </NavLink>
          ))}
        </nav>

        <div className="border-t border-zinc-800/80 p-3">
          <Button
            variant="ghost"
            size="md"
            className="w-full justify-start gap-3 text-zinc-400"
            onClick={signOut}
          >
            <LogOut size={18} />
            Sign Out
          </Button>
        </div>
      </aside>

      {/* Main content */}
      <div className="flex flex-1 flex-col overflow-hidden">
        <header className="flex h-14 items-center border-b border-zinc-800/80 px-4 lg:hidden">
          <button
            aria-label="Open menu"
            className="rounded-lg p-1.5 text-zinc-300 hover:bg-zinc-800"
            onClick={() => setSidebarOpen(true)}
          >
            <Menu size={20} />
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
