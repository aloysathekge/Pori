export type Role = "user" | "assistant";

export interface ToolChip {
  name: string;
  /** undefined while running, true/false once the tool_call_end lands. */
  success?: boolean;
}

export interface ChatMessage {
  id: string;
  role: Role;
  text: string;
  thinking: string;
  tools: ToolChip[];
  streaming: boolean;
}

export interface PendingClarification {
  id: string;
  question: string;
  options: string[];
}
