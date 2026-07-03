/**
 * `AloyClient` — the single typed client for the Aloy backend.
 *
 * Talks to the Pori/Aloy REST + SSE surface (`/v1/...`): submit a task, stream
 * its `PoriEvent`s, answer a structured clarification, and poll status/result.
 * Both the web and desktop surfaces consume this so the wire protocol lives in
 * one place. Auth is the `X-API-Key` header.
 */

import type { ClarificationRequestPayload, PoriEvent } from "./events";
import { parseSseStream } from "./sse";

export interface AloyClientOptions {
  /** Backend origin, e.g. "http://localhost:8000" (no trailing slash needed). */
  baseUrl: string;
  /** Sent as the `X-API-Key` header when present. */
  apiKey?: string;
  /** Injectable fetch (tests / non-browser runtimes). Defaults to global fetch. */
  fetch?: typeof fetch;
}

export interface TaskCreateResponse {
  task_id: string;
  status: string;
  submitted_at?: string;
}

export interface TaskStatusResponse {
  task_id: string;
  status: string;
  details?: string | null;
}

export interface TaskResultResponse {
  task_id: string;
  success: boolean;
  final_answer?: string | null;
  reasoning?: string | null;
}

export interface StreamHandlers {
  onEvent?: (event: PoriEvent) => void;
  onText?: (text: string, event: PoriEvent) => void;
  onThinking?: (text: string, event: PoriEvent) => void;
  onToolStart?: (event: PoriEvent) => void;
  onToolEnd?: (event: PoriEvent) => void;
  onClarification?: (
    request: ClarificationRequestPayload,
    event: PoriEvent,
  ) => void;
  onRunEnd?: (event: PoriEvent) => void;
  onError?: (error: unknown) => void;
}

export interface StreamTaskOptions {
  maxSteps?: number;
  signal?: AbortSignal;
}

export class AloyApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(`Aloy API error ${status}: ${message}`);
    this.name = "AloyApiError";
  }
}

export class AloyClient {
  private readonly baseUrl: string;
  private readonly apiKey?: string;
  private readonly doFetch: typeof fetch;

  constructor(options: AloyClientOptions) {
    this.baseUrl = options.baseUrl.replace(/\/+$/, "");
    this.apiKey = options.apiKey;
    this.doFetch = options.fetch ?? globalThis.fetch.bind(globalThis);
  }

  private headers(extra?: Record<string, string>): Record<string, string> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      ...extra,
    };
    if (this.apiKey) headers["X-API-Key"] = this.apiKey;
    return headers;
  }

  /** Submit a task to run in the background (no stream). */
  async submitTask(
    task: string,
    options: { maxSteps?: number } = {},
  ): Promise<TaskCreateResponse> {
    const res = await this.doFetch(`${this.baseUrl}/v1/tasks`, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify({ task, max_steps: options.maxSteps ?? 50 }),
    });
    if (!res.ok) throw new AloyApiError(res.status, await safeText(res));
    return (await res.json()) as TaskCreateResponse;
  }

  /**
   * Stream a task: `POST /v1/tasks/stream` (SSE over a POST body), dispatching
   * each `PoriEvent` to the handlers. Resolves when the stream closes; pass
   * `options.signal` to cancel.
   */
  async streamTask(
    task: string,
    handlers: StreamHandlers,
    options: StreamTaskOptions = {},
  ): Promise<void> {
    let res: Response;
    try {
      res = await this.doFetch(`${this.baseUrl}/v1/tasks/stream`, {
        method: "POST",
        headers: this.headers({ Accept: "text/event-stream" }),
        body: JSON.stringify({ task, max_steps: options.maxSteps ?? 50 }),
        signal: options.signal,
      });
    } catch (error) {
      handlers.onError?.(error);
      throw error;
    }
    if (!res.ok || !res.body) {
      const error = new AloyApiError(res.status, await safeText(res));
      handlers.onError?.(error);
      throw error;
    }
    try {
      for await (const event of parseSseStream(res.body, options.signal)) {
        dispatch(event, handlers);
      }
    } catch (error) {
      handlers.onError?.(error);
      throw error;
    }
  }

  /** Answer a structured clarification (the `ask_user` button bridge). */
  async submitClarification(
    clarificationId: string,
    value: string,
  ): Promise<void> {
    const res = await this.doFetch(
      `${this.baseUrl}/v1/clarify/${encodeURIComponent(clarificationId)}`,
      {
        method: "POST",
        headers: this.headers(),
        body: JSON.stringify({ value }),
      },
    );
    if (!res.ok) throw new AloyApiError(res.status, await safeText(res));
  }

  async getTaskStatus(taskId: string): Promise<TaskStatusResponse> {
    const res = await this.doFetch(
      `${this.baseUrl}/v1/tasks/${encodeURIComponent(taskId)}`,
      { headers: this.headers() },
    );
    if (!res.ok) throw new AloyApiError(res.status, await safeText(res));
    return (await res.json()) as TaskStatusResponse;
  }

  async getTaskResult(taskId: string): Promise<TaskResultResponse> {
    const res = await this.doFetch(
      `${this.baseUrl}/v1/tasks/${encodeURIComponent(taskId)}/result`,
      { headers: this.headers() },
    );
    if (!res.ok) throw new AloyApiError(res.status, await safeText(res));
    return (await res.json()) as TaskResultResponse;
  }
}

function dispatch(event: PoriEvent, handlers: StreamHandlers): void {
  handlers.onEvent?.(event);
  switch (event.type) {
    case "text_delta":
      handlers.onText?.((event.payload as { text?: string }).text ?? "", event);
      break;
    case "thinking_delta":
      handlers.onThinking?.(
        (event.payload as { text?: string }).text ?? "",
        event,
      );
      break;
    case "tool_call_start":
      handlers.onToolStart?.(event);
      break;
    case "tool_call_end":
      handlers.onToolEnd?.(event);
      break;
    case "clarification_request":
      handlers.onClarification?.(
        event.payload as unknown as ClarificationRequestPayload,
        event,
      );
      break;
    case "run_end":
      handlers.onRunEnd?.(event);
      break;
  }
}

async function safeText(res: Response): Promise<string> {
  try {
    return await res.text();
  } catch {
    return res.statusText;
  }
}
