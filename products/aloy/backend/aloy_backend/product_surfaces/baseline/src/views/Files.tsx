import { useState } from 'react';
import {
  ActionButton,
  openResource,
  useEventFiles,
  useSurfaceResourceState,
} from '@aloy/surface';
import { EmptyState, Section } from '../primitives';

export function FilesView() {
  const files = useEventFiles();
  const resource = useSurfaceResourceState('files');
  const [error, setError] = useState<string | null>(null);
  return (
    <Section heading="Files">
      <div {...resource.feedbackProps}>
        {files.length === 0 ? (
          <EmptyState
            status={resource.status}
            message="No files in this Event yet. Share one in the conversation and it appears here."
          />
        ) : (
          <ul className="baseline-list">
            {files.map((file) => (
              <li key={file.id}>
                <span className="baseline-list-title">{file.name}</span>
                <ActionButton
                  type="button"
                  variant="outline"
                  onClick={async () => {
                    setError(null);
                    try {
                      await openResource(file.id, { componentId: 'files' });
                    } catch {
                      setError(
                        `Aloy could not open ${file.name}. Try again from the Files view.`,
                      );
                    }
                  }}
                >
                  {`Open ${file.name}`}
                </ActionButton>
              </li>
            ))}
          </ul>
        )}
        {error ? <p role="alert">{error}</p> : null}
      </div>
    </Section>
  );
}
