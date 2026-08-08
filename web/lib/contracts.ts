export interface ClassProbability {
  denomination_pkr: number;
  probability: number;
}

export interface ClassificationResponse {
  request_id: string;
  model: { classifier_version: string; runtime: string };
  image: { width: number; height: number };
  prediction: {
    denomination_pkr: number;
    confidence: number;
    probabilities: ClassProbability[];
  };
  timings_ms: { decode: number; preprocess: number; inference: number; total: number };
  warnings: string[];
}

export interface ModelInfo {
  ready: boolean;
  model: { classifier_version: string; runtime: string };
  denominations: number[];
  preprocessing: Record<string, unknown>;
  artifact_checksums: Record<string, string>;
  quality_gate: { passed?: boolean; checks?: Record<string, boolean>; thresholds?: Record<string, number> };
  training_data_summary: Record<string, unknown>;
  evaluation_summary: Record<string, unknown>;
  limitations: string[];
  missing_artifacts: string[];
}

export interface ApiError {
  error?: { code: string; message: string; request_id: string; details: Record<string, unknown> };
}
