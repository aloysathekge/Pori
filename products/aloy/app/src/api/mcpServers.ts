import { apiFetch } from './client';
import type { ConnectionScope } from './connections';

export interface McpServerInfo {
  id: string;
  name: string;
  url: string;
  transport: string;
  auth_kind: string;
  scope: ConnectionScope;
  enabled: boolean;
  account_managed: boolean;
}

export interface McpServerCreate {
  name: string;
  url: string;
  transport?: 'http' | 'sse';
  auth_kind?: 'none' | 'static';
  static_secret?: string;
  scope?: ConnectionScope;
}

export function listMcpServers() {
  return apiFetch<McpServerInfo[]>('/mcp-servers');
}

export function createMcpServer(body: McpServerCreate) {
  return apiFetch<McpServerInfo>('/mcp-servers', {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

export function setMcpServerEnabled(id: string, enabled: boolean) {
  return apiFetch<McpServerInfo>(`/mcp-servers/${id}`, {
    method: 'PATCH',
    body: JSON.stringify({ enabled }),
  });
}

export function deleteMcpServer(id: string) {
  return apiFetch<void>(`/mcp-servers/${id}`, { method: 'DELETE' });
}
