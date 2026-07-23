import {
  useEventFiles,
  useSurfaceResourceState,
  useTasks,
} from '@aloy/surface';
import { EmptyState, Section, Stat } from '../primitives';

export function Overview() {
  const tasks = useTasks();
  const files = useEventFiles();
  const taskState = useSurfaceResourceState('tasks');
  const fileState = useSurfaceResourceState('files');
  const openTasks = tasks.filter((task) => task['status'] !== 'completed');
  return (
    <Section heading="Overview">
      <div className="baseline-stats" {...taskState.feedbackProps}>
        <Stat label="Open tasks" value={String(openTasks.length)} />
        <Stat label="Tasks in total" value={String(tasks.length)} />
        <Stat label="Files" value={String(files.length)} />
      </div>
      {tasks.length === 0 && files.length === 0 ? (
        <EmptyState
          status={taskState.status}
          message="This Event is just getting started. Add context in the conversation, upload files, or ask Aloy what to do first."
        />
      ) : (
        <p {...fileState.feedbackProps}>
          Use the Tasks and Files views for detail, or ask Aloy below to change
          how this workspace looks.
        </p>
      )}
    </Section>
  );
}
