import { apiFetch } from './client';
import type { UsageSummary, UsageHistoryEntry, UsageRecord } from '@/types';

export function getUsageSummary(days = 30) {
  return apiFetch<UsageSummary>(`/me/usage?days=${days}`);
}

export function getUsageHistory(days = 30) {
  return apiFetch<UsageHistoryEntry[]>(`/me/usage/history?days=${days}`);
}

export function getUsageRecords(limit = 50, offset = 0) {
  return apiFetch<UsageRecord[]>(
    `/me/usage/records?limit=${limit}&offset=${offset}`,
  );
}
