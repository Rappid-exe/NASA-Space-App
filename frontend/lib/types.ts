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

export interface FeatureImportance {
  algorithm: string;
  features: Array<{
    name: string;
    importance: number;
    rank: number;
  }>;
  top_5: Array<{
    name: string;
    importance: number;
  }>;
}

export interface ExoplanetEducation {
  overview: string;
  detection_methods: Record<string, any>;
  planet_types: Record<string, any>;
  features_explained: Record<string, string>;
  missions: Record<string, any>;
}

export interface DatasetComparison {
  missions: Array<{
    name: string;
    total_observations: number;
    confirmed_exoplanets: number;
    false_positives: number;
    candidates: number;
    confirmation_rate: number;
  }>;
  summary: {
    total_observations: number;
    total_confirmed: number;
    total_false_positives: number;
  };
}

export interface HyperparameterTuningRequest {
  algorithm: string;
  param_grid: Record<string, any[]>;
  cv_folds?: number;
}

export interface HyperparameterTuningResult {
  best_params: Record<string, any>;
  best_score: number;
  cv_results: {
    mean_scores: number[];
    std_scores: number[];
  };
}

export interface RetrainingRequest {
  algorithm: string;
  dataset?: string;
  hyperparameters?: Record<string, any>;
}

export interface RetrainingResult {
  message: string;
  model_id: string;
  algorithm: string;
  dataset: string;
  performance: Record<string, any>;
}
