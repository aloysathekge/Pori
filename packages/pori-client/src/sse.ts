/**
 * Incremental SSE decoder for the Pori event stream.
 *
 * The backend streams `text/event-stream` frames of the form
 * `event: <type>\ndata: {"payload": ..., "step": N}\n\n`. Because the stream
 * is served over a POST body (not a GET, so `EventSource` can't be used), we
 * read the `fetch` `ReadableStream` and parse frames by hand. Keepalive
 * comments (`: ...`) and unknown fields are ignored.
 */

import type { PoriEvent } from "./events";

export async function* parseSseStream(
  body: ReadableStream<Uint8Array>,
  signal?: AbortSignal,
): AsyncGenerator<PoriEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      if (signal?.aborted) break;
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let boundary: number;
      while ((boundary = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const event = parseFrame(frame);
        if (event) yield event;
      }
    }
  } finally {
    reader.releaseLock();
  }
}

export function parseFrame(frame: string): PoriEvent | null {
  let eventType = "message";
  const dataLines: string[] = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith(":")) continue; // keepalive comment
    if (line.startsWith("event:")) eventType = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (dataLines.length === 0) return null;
  try {
    const parsed = JSON.parse(dataLines.join("\n")) as {
      payload?: Record<string, unknown>;
      step?: number;
    };
    return { type: eventType, payload: parsed.payload ?? {}, step: parsed.step ?? 0 };
  } catch {
    return null;
  }
}
