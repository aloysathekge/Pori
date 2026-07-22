import { apiFetch } from './client';
import type { EventSummary } from './events';

export type EventTemplateDiscoveryGroup =
  | 'student'
  | 'individual'
  | 'professional'
  | 'team'
  | 'business'
  | string;

export interface EventTemplateSummary {
  id: string;
  slug: string;
  title: string;
  summary: string;
  discovery_group: EventTemplateDiscoveryGroup;
  current_release: {
    id: string;
    version: number;
    schema_version: number;
  };
  updated_at: string;
}

export interface EventTemplateDetail extends EventTemplateSummary {
  release: {
    id: string;
    version: number;
    schema_version: number;
    release_notes: string;
    checksum: string;
    compatibility: Array<{
      key: string;
      requirement: unknown;
      required: boolean;
    }>;
    assets: Array<{
      key: string;
      kind: string;
      content_type: string;
      sha256: string;
      size_bytes: number;
    }>;
    guided_jobs: Array<{
      key: string;
      title: string;
      priority: string;
      materializes_task: boolean;
    }>;
  };
}

export interface EventTemplateInstallation {
  installation: {
    id: string;
    template_id: string;
    release_id: string;
    event_id: string;
    status: string;
    installed_at: string;
  };
  event: EventSummary;
  surface: {
    project_id: string | null;
    status: 'not_seeded' | 'published' | 'preparing' | 'failed' | 'source_seeded';
    run_id: string | null;
    run_status: string | null;
  };
  replayed: boolean;
}

export function listEventTemplates() {
  return apiFetch<{ templates: EventTemplateSummary[] }>('/event-templates');
}

export function getEventTemplate(templateId: string) {
  return apiFetch<EventTemplateDetail>(`/event-templates/${templateId}`);
}

export function installEventTemplate(
  templateId: string,
  input: {
    idempotency_key: string;
    release_id: string;
    title?: string;
  },
) {
  return apiFetch<EventTemplateInstallation>(`/event-templates/${templateId}/install`, {
    method: 'POST',
    body: JSON.stringify(input),
  });
}
