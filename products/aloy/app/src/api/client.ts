const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/v1';

let getToken: (() => Promise<string | null>) | null = null;

export function setTokenGetter(fn: () => Promise<string | null>) {
  getToken = fn;
}

async function authHeaders(): Promise<Record<string, string>> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (getToken) {
    const token = await getToken();
    if (token) headers['Authorization'] = `Bearer ${token}`;
  }
  return headers;
}

export async function apiFetch<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const headers = await authHeaders();
  const res = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers: { ...headers, ...(options.headers as Record<string, string>) },
  });

  if (res.status === 204) return undefined as T;

  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, body.detail || 'Unknown error');
  }

  return res.json() as Promise<T>;
}

export async function apiTextFetch(
  path: string,
  options: RequestInit = {},
): Promise<string> {
  const headers = await authHeaders();
  const res = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers: { ...headers, ...(options.headers as Record<string, string>) },
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, body.detail || 'Unknown error');
  }
  return res.text();
}

export async function apiBlobFetch(
  path: string,
  options: RequestInit = {},
): Promise<Blob> {
  const headers = await authHeaders();
  const res = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers: { ...headers, ...(options.headers as Record<string, string>) },
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, body.detail || 'Unknown error');
  }
  return res.blob();
}

export async function apiStreamFetch(
  path: string,
  body?: unknown,
  signal?: AbortSignal,
  method: 'POST' | 'GET' = 'POST',
): Promise<Response> {
  const headers = await authHeaders();
  const res = await fetch(`${BASE_URL}${path}`, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
    signal,
  });

  if (!res.ok) {
    const errBody = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, errBody.detail || 'Stream error');
  }

  return res;
}

/** Multipart upload with progress (XHR — fetch can't observe upload bytes). */
export async function apiUploadFile<T>(
  path: string,
  file: File,
  onProgress?: (pct: number) => void,
): Promise<T> {
  const token = getToken ? await getToken() : null;
  return new Promise<T>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${BASE_URL}${path}`);
    if (token) xhr.setRequestHeader('Authorization', `Bearer ${token}`);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText) as T);
      } else {
        let detail = 'Upload failed';
        try {
          detail = JSON.parse(xhr.responseText).detail || detail;
        } catch {
          // keep the fallback message
        }
        reject(new ApiError(xhr.status, detail));
      }
    };
    xhr.onerror = () => reject(new ApiError(0, 'Network error during upload'));
    const form = new FormData();
    form.append('file', file);
    xhr.send(form);
  });
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = 'ApiError';
  }
}
