import type { PartListResponse, PredictionResponse } from '../types/prediction';

const PREDICT_ENDPOINT = '/predict';
const PARTS_ENDPOINT = '/parts';

export class PredictionApiError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'PredictionApiError';
  }
}

/**
 * Fetch the list of available trained part types from the backend.
 * Returns an empty array if none are trained yet.
 */
export async function fetchAvailableParts(): Promise<string[]> {
  try {
    const response = await fetch(PARTS_ENDPOINT);
    if (!response.ok) return [];
    const data = (await response.json()) as PartListResponse;
    return data.parts ?? [];
  } catch {
    return [];
  }
}

/**
 * Submit an image for anomaly detection.
 *
 * @param file      - The image file to analyse.
 * @param partType  - Which part model to use (e.g. "part_A"). Required.
 */
export async function predictImage(
  file: File,
  partType: string,
): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const url = `${PREDICT_ENDPOINT}?part_type=${encodeURIComponent(partType)}`;

  let response: Response;
  try {
    response = await fetch(url, {
      method: 'POST',
      body: formData,
    });
  } catch {
    throw new PredictionApiError('Failed to analyze image');
  }

  if (!response.ok) {
    let detail = 'Failed to analyze image';
    try {
      const body = await response.json();
      if (body?.detail) detail = String(body.detail);
    } catch { /* ignore */ }
    throw new PredictionApiError(detail);
  }

  return response.json() as Promise<PredictionResponse>;
}
