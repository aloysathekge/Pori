import { createContext, useContext, type Dispatch, type SetStateAction } from 'react';

interface WorkspaceFocusValue {
  focused: boolean;
  setFocused: Dispatch<SetStateAction<boolean>>;
}

export const WorkspaceFocusContext = createContext<WorkspaceFocusValue | null>(null);

export function useWorkspaceFocus(): WorkspaceFocusValue {
  const value = useContext(WorkspaceFocusContext);
  if (!value) throw new Error('Workspace focus must be used inside AppLayout');
  return value;
}
