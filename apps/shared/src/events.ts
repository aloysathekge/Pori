/**
 * The Aloy event contract — a 1:1 mirror of the Pori kernel's `PoriEvent`
 * (`pori/observability/events.py`). Every backend SSE frame decodes to one of
 * these; every surface renders from this single shape.
 */

export const RUN_START = "run_start";
export const RUN_END = "run_end";
export const STEP_START = "step_start";
export const STEP_END = "step_end";
export const TEXT_DELTA = "text_delta"; // visible answer prose, streamed
export const THINKING_DELTA = "thinking_delta"; // reasoning prose, streamed
export const TOOL_CALL_START = "tool_call_start"; // tool name known
export const TOOL_CALL_END = "tool_call_end"; // after execution: success + result
export const LLM_RETRY = "llm_retry"; // API retrying / rate-limited
export const CLARIFICATION_REQUEST = "clarification_request"; // ask_user w/ options

export type PoriEventType =
  | typeof RUN_START
  | typeof RUN_END
  | typeof STEP_START
  | typeof STEP_END
  | typeof TEXT_DELTA
  | typeof THINKING_DELTA
  | typeof TOOL_CALL_START
  | typeof TOOL_CALL_END
  | typeof LLM_RETRY
  | typeof CLARIFICATION_REQUEST;

/** One normalized agent event: `{ type, payload, step }`. */
export interface PoriEvent<P = Record<string, unknown>> {
  type: PoriEventType | string;
  payload: P;
  step: number;
}

// --- payload shapes (event-specific) ----------------------------------------
export interface TextDeltaPayload {
  text: string;
}

export interface ToolCallStartPayload {
  name: string;
  [key: string]: unknown;
}

export interface ToolCallEndPayload {
  name?: string;
  success?: boolean;
  result?: unknown;
  [key: string]: unknown;
}

/** A structured `ask_user` request the UI renders as buttons; answer it via
 *  `AloyClient.submitClarification(id, value)`. */
export interface ClarificationRequestPayload {
  type: typeof CLARIFICATION_REQUEST;
  id: string;
  question: string;
  options: string[];
}
