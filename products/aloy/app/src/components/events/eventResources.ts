import type { EventFile } from '@/api/events';

export interface EventResourceGroups {
  sources: EventFile[];
  artifacts: EventFile[];
}

export function groupEventResources(files: EventFile[]): EventResourceGroups {
  return files.reduce<EventResourceGroups>(
    (groups, file) => {
      if (file.kind === 'artifact') groups.artifacts.push(file);
      else groups.sources.push(file);
      return groups;
    },
    { sources: [], artifacts: [] },
  );
}
