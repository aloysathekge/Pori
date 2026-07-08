import { apiFetch } from './client';

export type ConnectionScope = 'user' | 'org';

export interface ProviderInfo {
  provider: string;
  label: string;
  description: string;
  connected: boolean;
  status: string | null;
  account_email: string | null;
  org_connected: boolean;
  org_status: string | null;
  org_account_email: string | null;
  can_manage_org: boolean;
}

export function listConnections() {
  return apiFetch<ProviderInfo[]>('/connections');
}

export function startConnection(provider: string, scope: ConnectionScope = 'user') {
  return apiFetch<{ authorize_url: string }>(
    `/connections/${provider}/start?scope=${scope}`,
    { method: 'POST' },
  );
}

export function disconnect(provider: string, scope: ConnectionScope = 'user') {
  return apiFetch<{ provider: string; connected: boolean }>(
    `/connections/${provider}?scope=${scope}`,
    { method: 'DELETE' },
  );
}
