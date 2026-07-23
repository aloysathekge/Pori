import { useState } from 'react';
import { SurfaceRoot, useSurfaceContext } from '@aloy/surface';
import { AskAloy } from './views/AskAloy';
import { FilesView } from './views/Files';
import { Overview } from './views/Overview';
import { TasksView } from './views/Tasks';

type ViewName = 'overview' | 'tasks' | 'files';

interface EventSummary {
  title?: string;
  summary?: string;
  phase?: string;
}

export default function App() {
  const context = useSurfaceContext();
  const event = (context?.data.event ?? {}) as EventSummary;
  const [view, setView] = useState<ViewName>('overview');
  return (
    <SurfaceRoot>
      <header className="baseline-header">
        <h1>{event.title || 'This Event'}</h1>
        <p>
          {event.summary ||
            "Aloy keeps this Event's work, files, and plans in one place. " +
              'Ask Aloy to reshape this workspace at any time.'}
        </p>
      </header>
      <nav aria-label="Surface views" className="baseline-nav">
        <button
          type="button"
          aria-pressed={view === 'overview'}
          onClick={() => setView('overview')}
        >
          Overview
        </button>
        <button
          type="button"
          aria-pressed={view === 'tasks'}
          onClick={() => setView('tasks')}
        >
          Tasks
        </button>
        <button
          type="button"
          aria-pressed={view === 'files'}
          onClick={() => setView('files')}
        >
          Files
        </button>
      </nav>
      {view === 'overview' ? (
        <Overview />
      ) : view === 'tasks' ? (
        <TasksView />
      ) : (
        <FilesView />
      )}
      <AskAloy />
    </SurfaceRoot>
  );
}
