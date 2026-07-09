// ---- Conversations ----
export interface ConversationResponse {
  id: string;
  title: string | null;
  agent_config_id: string | null;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface MessageImage {
  data: string; // base64 (no data: prefix)
  media_type: string;
}

export interface MessageFile {
  name: string;
  size: number;
  content?: string; // present on persisted messages (context rebuild)
}

export interface MessageMetadata {
  images?: MessageImage[];
  files?: MessageFile[];
  reasoning?: string | null;
  steps_taken?: number;
  metrics?: Record<string, unknown> | null;
  artifacts?: Artifact[];
  plan?: PlanItem[];
  selected_skills?: string[];
  run_id?: string | null;
}

export interface MessageResponse {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  metadata: MessageMetadata | null;
  created_at: string;
}

export interface ConversationDetail {
  id: string;
  title: string | null;
  agent_config_id: string | null;
  created_at: string;
  updated_at: string;
  messages: MessageResponse[];
}

export interface SendMessageRequest {
  content: string;
  max_steps?: number;
  stream?: boolean;
  team_id?: string | null;
}

// ---- SOUL (identity) ----
export interface SoulResponse {
  content: string;
  char_limit: number;
  updated_at?: string | null;
}

// ---- SSE Events ----
export type PlanItemStatus =
  | 'pending'
  | 'in_progress'
  | 'completed'
  | 'cancelled';

export interface PlanItem {
  id: string;
  content: string;
  status: PlanItemStatus;
}

export interface Artifact {
  kind?: string;
  tool_name?: string;
  path?: string;
  operation?: string;
  bytes_written?: number;
  receipt_id?: string;
}

export interface SSEStatusEvent {
  status: string;
  task: string;
}

/** One tool call as it happens (full activity log). */
export interface SSEToolEvent {
  step: number;
  tool: string;
  preview: string;
  success: boolean;
  args?: Record<string, unknown>;
  result?: unknown;
}

/** A step boundary: model intent line + live plan checklist. */
export interface SSEStepEvent {
  step: number;
  max_steps: number;
  activity?: string;
  plan?: PlanItem[];
}

export interface SSEMessageEvent {
  role: string;
  run_id?: string | null;
  content: string;
  reasoning?: string;
  steps_taken?: number;
  success?: boolean;
  metrics?: Record<string, unknown>;
  selected_skills?: string[];
  artifacts?: Artifact[];
  plan?: PlanItem[];
}

// ---- Runs ----
export type RunStatus = 'pending' | 'running' | 'completed' | 'failed';

export interface RunResponse {
  id: string;
  status: RunStatus;
  success: boolean;
  steps_taken: number;
  final_answer: string | null;
  reasoning: string | null;
  metrics: Record<string, unknown> | null;
  created_at: string;
}

// ---- Agent Configs ----
export type Provider = 'anthropic' | 'openai' | 'google';

export interface AgentConfigCreate {
  name: string;
  provider: Provider;
  model: string;
  temperature?: number;
  max_steps?: number;
  system_prompt?: string | null;
  tools?: string[] | null;
  is_default?: boolean;
}

export interface AgentConfigResponse {
  id: string;
  name: string;
  provider: string;
  model: string;
  temperature: number;
  max_steps: number;
  system_prompt: string | null;
  tools: string[] | null;
  is_default: boolean;
  created_at: string;
}

export interface ToolInfo {
  name: string;
  description: string;
}

// ---- Teams ----
export type TeamMode = 'router' | 'broadcast' | 'delegate';

export interface TeamMember {
  name: string;
  description: string;
  llm_config?: { provider: string; model: string } | null;
  agent_settings?: { max_steps?: number } | null;
  tools?: string[] | null;
}

export interface TeamConfigCreate {
  name: string;
  mode: TeamMode;
  members: TeamMember[];
  max_delegation_steps?: number;
  max_concurrent_members?: number;
}

export interface TeamConfigResponse {
  id: string;
  name: string;
  mode: TeamMode;
  members: TeamMember[];
  max_delegation_steps: number;
  max_concurrent_members: number;
  created_at: string;
}

export interface TeamRunResponse {
  task: string;
  completed: boolean;
  steps_taken: number;
  final_answer: string;
  mode: string;
  metrics: Record<string, unknown> | null;
}

// ---- User Profile ----
export interface UserProfileResponse {
  id: string;
  display_name: string | null;
  avatar_url: string | null;
  default_agent_config_id: string | null;
  preferences: Record<string, unknown> | null;
  created_at: string;
  updated_at: string;
}

export interface UserProfileUpdate {
  display_name?: string;
  avatar_url?: string;
  default_agent_config_id?: string;
  preferences?: Record<string, unknown>;
}

export interface UsageStatsResponse {
  total_conversations: number;
  total_messages: number;
  total_runs: number;
  member_since: string;
}

// ---- Memory ----
export interface MemoryBlock {
  label: string;
  value: string;
  limit: number;
  read_only: boolean;
}

export type KnowledgeKind = 'semantic' | 'episodic' | 'procedural';

export interface KnowledgeEntry {
  id: string;
  organization_id?: string;
  user_id?: string;
  agent_id?: string | null;
  session_id?: string | null;
  content: string;
  tags: string[] | null;
  importance: number;
  kind?: KnowledgeKind;
  confidence?: number;
  sensitivity?: string;
  source: string;
  status?: string;
  created_at: string | null;
  updated_at?: string | null;
  event_at?: string | null;
}

export interface KnowledgeEntryCreate {
  content: string;
  tags?: string[] | null;
  importance?: number;
  kind?: KnowledgeKind;
  source?: string;
}

export interface KnowledgeSearchRequest {
  query: string;
  k?: number;
  tags?: string[] | null;
  kinds?: KnowledgeKind[] | null;
  min_score?: number;
}

/** @deprecated Use KnowledgeEntry — archival routes are API aliases */
export type ArchivalPassage = KnowledgeEntry;

// ---- Usage & Billing ----
export interface UsageSummary {
  total_tokens: number;
  total_cost: number;
  total_requests: number;
  by_model: Record<string, { tokens: number; cost: number; requests: number }>;
}

export interface UsageHistoryEntry {
  date: string;
  tokens: number;
  cost: number;
  requests: number;
}

export interface UsageRecord {
  id: string;
  run_id: string | null;
  conversation_id: string | null;
  provider: string;
  model: string;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  estimated_cost: number;
  created_at: string;
}

// ---- Traces ----
export interface TraceListItem {
  id: string;
  run_id: string | null;
  conversation_id: string | null;
  duration_seconds: number;
  total_spans: number;
  status: string;
  created_at: string;
}

export interface TraceSpan {
  span_id: string;
  name: string;
  type?: string;
  status: string;
  duration: string;
  attributes?: Record<string, unknown>;
  children: TraceSpan[];
}

export interface TraceData {
  trace_id: string;
  name: string;
  run_id: string;
  status: string;
  duration: string;
  total_spans: number;
  input: string;
  output: string;
  tree: TraceSpan[];
}

export interface TraceDetail {
  id: string;
  run_id: string;
  conversation_id: string;
  trace_data: TraceData;
  duration_seconds: number;
  total_spans: number;
  status: string;
  created_at: string;
}
