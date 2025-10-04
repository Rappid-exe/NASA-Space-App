// API Types for Exoplanet Classification

export interface ExoplanetFeatures {
  orbital_period: number;
  transit_duration: number;
  transit_depth: number;
  planetary_radius: number;
  equilibrium_temperature?: number;
}

export interface ClassificationResult {
  prediction: 'CONFIRMED' | 'FALSE_POSITIVE';
  confidence: number;
  probabilities: {
    FALSE_POSITIVE: number;
    CONFIRMED: number;
  };
  explanation: string;
}

export interface BatchClassificationRequest {
  observations: ExoplanetFeatures[];
}

export interface BatchClassificationResponse {
  results: ClassificationResult[];
  total_processed: number;
  summary: {
    CONFIRMED: number;
    FALSE_POSITIVE: number;
  };
}

export interface ModelStatistics {
  model_id: string;
  model_name: string;
  algorithm: string;
  version: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  training_date: string;
  training_samples: number;
  test_samples: number;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_id: string | null;
}

export interface ModelInfo {
  model_id: string;
  version: number;
  algorithm: string;
  f1_score: number;
  accuracy: number;
  training_date: string;
}
