import { useSurfaceResourceState, useTasks } from '@aloy/surface';
import { EmptyState, Section } from '../primitives';

export function TasksView() {
  const tasks = useTasks();
  const resource = useSurfaceResourceState('tasks');
  return (
    <Section heading="Tasks">
      <div {...resource.feedbackProps}>
        {tasks.length === 0 ? (
          <EmptyState
            status={resource.status}
            message="No tasks in this Event yet."
          />
        ) : (
          <ul className="baseline-list">
            {tasks.map((task, index) => (
              <li key={String(task['id'] ?? index)}>
                <span className="baseline-list-title">
                  {String(task['title'] ?? 'Task')}
                </span>
                <span className="baseline-badge">
                  {String(task['status'] ?? 'planned')}
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </Section>
  );
}
