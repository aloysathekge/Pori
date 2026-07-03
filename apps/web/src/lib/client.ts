import { AloyClient } from "@aloy/shared";

const baseUrl = import.meta.env.VITE_ALOY_API_URL ?? "http://localhost:8000";

function readApiKey(): string | undefined {
  if (import.meta.env.VITE_ALOY_API_KEY) return import.meta.env.VITE_ALOY_API_KEY;
  if (typeof localStorage !== "undefined") {
    return localStorage.getItem("aloy_api_key") ?? undefined;
  }
  return undefined;
}

/** The single backend client for the web surface. */
export const client = new AloyClient({ baseUrl, apiKey: readApiKey() });
