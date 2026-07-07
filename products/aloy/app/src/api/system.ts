import { apiFetch } from './client';

export interface ExecutionStatus {
  enabled: boolean;
  backend: string;
  isolated: boolean;
  label: string;
  detail: string;
}

export function getExecutionStatus() {
  return apiFetch<ExecutionStatus>('/system/execution');
}
