import { apiFetch } from './client';

export interface SkillResponse {
  id: string;
  slug: string;
  version: string;
  name: string;
  summary: string;
  instructions: string;
  tags: string[];
  category: string;
  status: string;
  created_at: string;
  updated_at: string;
}

export interface SkillCreate {
  slug: string;
  version?: string;
  name: string;
  summary: string;
  instructions: string;
  tags?: string[];
  category?: string;
}

export type SkillUpdate = Partial<Omit<SkillCreate, 'slug'>>;

export function listSkills() {
  return apiFetch<SkillResponse[]>('/skills');
}

export function getSkill(id: string) {
  return apiFetch<SkillResponse>(`/skills/${id}`);
}

export function createSkill(data: SkillCreate) {
  return apiFetch<SkillResponse>('/skills', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export function updateSkill(id: string, data: SkillUpdate) {
  return apiFetch<SkillResponse>(`/skills/${id}`, {
    method: 'PATCH',
    body: JSON.stringify(data),
  });
}
