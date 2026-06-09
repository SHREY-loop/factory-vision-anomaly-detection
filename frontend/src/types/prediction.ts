/**
 * Type definitions for the anomaly detection inspection API.
 *
 * Phase 1 fields are always populated.
 * Phase 2+ fields are optional — present in the schema for extensibility
 * but return null/[] until YOLO / Grad-CAM are implemented.
 */

export type InspectionStatus = 'OK' | 'NOT_OK';

/** Raw API response from POST /predict */
export interface PredictionResponse {
  // Phase 1 — Anomaly detection core (always present)
  status: InspectionStatus;
  confidence: number;       // 0-100
  anomaly_score: number;    // 0.0-1.0 normalised

  // Diagnostics (always present)
  raw_distance: number;     // L2 distance to nearest OK embedding
  threshold: number;        // calibrated decision boundary (same units as raw_distance)

  // Part type routing
  part_type: string | null; // which model was used

  // Phase 2 — YOLO defect localisation (future, currently null/[])
  defect_type?: string | null;
  bounding_boxes?: unknown[];

  // Phase 3 — Explainability (future, currently null)
  heatmap?: string | null;
}

/** Response from GET /parts */
export interface PartListResponse {
  parts: string[];
}

/** Transformed result for display in UI components */
export interface PredictionDisplay {
  statusLabel: string;
  confidence: number;   // 0-100
  anomaly_score: number; // 0.0-1.0
  raw_distance: number;    // L2 distance
  threshold: number;       // decision boundary
  isOk: boolean;
  part_type: string | null;
}
