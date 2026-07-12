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

export interface SkillImportPreview extends SkillCreate {
  version: string;
  tags: string[];
  category: string;
  author: string;
  license: string;
  warnings: string[];
}

/** Parse a SKILL.md (paste a URL or the file text) into prefilled create
 *  fields — import-first UX; nobody hand-types slugs. */
export function previewSkillImport(input: { url?: string; text?: string }) {
  return apiFetch<SkillImportPreview>('/skills/preview', {
    method: 'POST',
    body: JSON.stringify(input),
  });
}

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
