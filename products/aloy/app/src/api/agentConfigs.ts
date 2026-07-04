import { apiFetch } from './client';
import type { AgentConfigCreate, AgentConfigResponse, ToolInfo } from '@/types';

export function listAgentConfigs() {
  return apiFetch<AgentConfigResponse[]>('/agent-configs');
}

export function getAgentConfig(id: string) {
  return apiFetch<AgentConfigResponse>(`/agent-configs/${id}`);
}

export function createAgentConfig(data: AgentConfigCreate) {
  return apiFetch<AgentConfigResponse>('/agent-configs', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export function updateAgentConfig(id: string, data: Partial<AgentConfigCreate>) {
  return apiFetch<AgentConfigResponse>(`/agent-configs/${id}`, {
    method: 'PATCH',
    body: JSON.stringify(data),
  });
}

export function deleteAgentConfig(id: string) {
  return apiFetch<void>(`/agent-configs/${id}`, { method: 'DELETE' });
}

export function getAvailableModels() {
  return apiFetch<Record<string, string[]>>('/agent-configs/info/models');
}

export function getAvailableTools() {
  return apiFetch<ToolInfo[]>('/agent-configs/info/tools');
}
