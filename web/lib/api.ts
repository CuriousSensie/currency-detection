import type { ApiError, ClassificationResponse, ModelInfo } from "./contracts";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function parseError(response: Response): Promise<Error> {
  const body = (await response.json().catch(() => ({}))) as ApiError;
  return new Error(body.error?.message ?? `Analysis service returned HTTP ${response.status}.`);
}

export async function fetchModelInfo(signal?: AbortSignal): Promise<ModelInfo> {
  const response = await fetch(`${API_URL}/v1/model-info`, { signal, cache: "no-store" });
  if (!response.ok) throw await parseError(response);
  return response.json() as Promise<ModelInfo>;
}

export async function classifyImage(file: File, signal?: AbortSignal): Promise<ClassificationResponse> {
  const form = new FormData();
  form.append("image", file, file.name);
  const response = await fetch(`${API_URL}/v1/classify`, {
    method: "POST",
    body: form,
    signal,
  });
  if (!response.ok) throw await parseError(response);
  return response.json() as Promise<ClassificationResponse>;
}
