import { apiFetch } from './client';
import type { UserProfileResponse, UserProfileUpdate } from '@/types';

export function getProfile() {
  return apiFetch<UserProfileResponse>('/me');
}

export function updateProfile(data: UserProfileUpdate) {
  return apiFetch<UserProfileResponse>('/me', {
    method: 'PATCH',
    body: JSON.stringify(data),
  });
}
