import { apiFetch } from './client';

export interface ProviderInfo {
  provider: string;
  label: string;
  description: string;
  connected: boolean;
  status: string | null;
  account_email: string | null;
}

export function listConnections() {
  return apiFetch<ProviderInfo[]>('/connections');
}

export function startConnection(provider: string) {
  return apiFetch<{ authorize_url: string }>(`/connections/${provider}/start`, {
    method: 'POST',
  });
}

export function disconnect(provider: string) {
  return apiFetch<{ provider: string; connected: boolean }>(
    `/connections/${provider}`,
    { method: 'DELETE' },
  );
}
