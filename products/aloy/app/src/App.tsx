import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from '@/contexts/AuthContext';
import { useAuth } from '@/contexts/useAuth';
import { AppLayout } from '@/components/layout/AppLayout';
import { LoginPage } from '@/pages/LoginPage';
import { SignupPage } from '@/pages/SignupPage';
import { ChatPage } from '@/pages/ChatPage';
import { AgentConfigsPage } from '@/pages/AgentConfigsPage';
import { FilesPage } from '@/pages/FilesPage';
import { SkillsPage } from '@/pages/SkillsPage';
import { SchedulesPage } from '@/pages/SchedulesPage';
import { ConnectionsPage } from '@/pages/ConnectionsPage';
import { TeamsPage } from '@/pages/TeamsPage';
import { MemoryPage } from '@/pages/MemoryPage';
import { UsagePage } from '@/pages/UsagePage';
import { TracesPage } from '@/pages/TracesPage';
import { SettingsPage } from '@/pages/SettingsPage';
import { TodayPage } from '@/pages/TodayPage';
import { EventPage } from '@/pages/EventPage';
import { EventSetupPage } from '@/pages/EventSetupPage';
import { Spinner } from '@/components/ui/Spinner';
import type { ReactNode } from 'react';

function ProtectedRoute({ children }: { children: ReactNode }) {
  const { session, loading } = useAuth();

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center bg-zinc-950">
        <Spinner className="h-8 w-8" />
      </div>
    );
  }

  if (!session) return <Navigate to="/login" replace />;
  return <>{children}</>;
}

function PublicRoute({ children }: { children: ReactNode }) {
  const { session, loading } = useAuth();

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center bg-zinc-950">
        <Spinner className="h-8 w-8" />
      </div>
    );
  }

  if (session) return <Navigate to="/today" replace />;
  return <>{children}</>;
}

function AppRoutes() {
  return (
    <Routes>
      <Route
        path="/login"
        element={
          <PublicRoute>
            <LoginPage />
          </PublicRoute>
        }
      />
      <Route
        path="/signup"
        element={
          <PublicRoute>
            <SignupPage />
          </PublicRoute>
        }
      />
      <Route
        path="/"
        element={
          <ProtectedRoute>
            <AppLayout />
          </ProtectedRoute>
        }
      >
        <Route index element={<Navigate to="/today" replace />} />
        <Route path="today" element={<TodayPage />} />
        <Route path="events/new" element={<EventSetupPage />} />
        <Route path="events/:eventId" element={<EventPage />} />
        <Route path="chat" element={<ChatPage />} />
        <Route path="chat/:conversationId" element={<ChatPage />} />
        <Route path="agents" element={<AgentConfigsPage />} />
        <Route path="files" element={<FilesPage />} />
        <Route path="skills" element={<SkillsPage />} />
        <Route path="schedules" element={<SchedulesPage />} />
        <Route path="connections" element={<ConnectionsPage />} />
        <Route path="teams" element={<TeamsPage />} />
        <Route path="memory" element={<MemoryPage />} />
        <Route path="usage" element={<UsagePage />} />
        <Route path="traces" element={<TracesPage />} />
        <Route path="settings" element={<SettingsPage />} />
      </Route>
      <Route path="*" element={<Navigate to="/today" replace />} />
    </Routes>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <AppRoutes />
      </AuthProvider>
    </BrowserRouter>
  );
}
