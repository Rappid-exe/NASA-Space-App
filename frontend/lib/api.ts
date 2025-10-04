// API Client for Exoplanet Classification Backend

import axios from 'axios';
import type {
  ExoplanetFeatures,
  ClassificationResult,
  BatchClassificationRequest,
  BatchClassificationResponse,
  ModelStatistics,
  HealthResponse,
  ModelInfo,
  FeatureImportance,
  ExoplanetEducation,
  DatasetComparison,
  HyperparameterTuningRequest,
  HyperparameterTuningResult,
  RetrainingRequest,
  RetrainingResult,
} from './types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 seconds
});

// Health Check
export async function checkHealth(): Promise<HealthResponse> {
  const response = await apiClient.get<HealthResponse>('/health');
  return response.data;
}

// Single Classification
export async function classifyObservation(
  features: ExoplanetFeatures
): Promise<ClassificationResult> {
  const response = await apiClient.post<ClassificationResult>('/classify', features);
  return response.data;
}

// Batch Classification
export async function classifyBatch(
  request: BatchClassificationRequest
): Promise<BatchClassificationResponse> {
  const response = await apiClient.post<BatchClassificationResponse>(
    '/classify/batch',
    request
  );
  return response.data;
}

// Model Statistics
export async function getModelStatistics(): Promise<ModelStatistics> {
  const response = await apiClient.get<ModelStatistics>('/model/statistics');
  return response.data;
}

// List Models
export async function listModels(): Promise<ModelInfo[]> {
  const response = await apiClient.get<ModelInfo[]>('/models/list');
  return response.data;
}

// Load Specific Model
export async function loadModel(modelId: string): Promise<{ message: string; model_id: string; algorithm: string }> {
  const response = await apiClient.post(`/model/load/${modelId}`);
  return response.data;
}

// Feature Importance
export async function getFeatureImportance(): Promise<FeatureImportance> {
  const response = await apiClient.get<FeatureImportance>('/model/feature-importance');
  return response.data;
}

// Educational Content
export async function getExoplanetEducation(): Promise<ExoplanetEducation> {
  const response = await apiClient.get<ExoplanetEducation>('/education/exoplanet-info');
  return response.data;
}

// Dataset Comparison
export async function getDatasetComparison(): Promise<DatasetComparison> {
  const response = await apiClient.get<DatasetComparison>('/datasets/comparison');
  return response.data;
}

// Hyperparameter Tuning
export async function tuneHyperparameters(
  request: HyperparameterTuningRequest
): Promise<HyperparameterTuningResult> {
  const response = await apiClient.post<HyperparameterTuningResult>(
    '/model/tune-hyperparameters',
    request
  );
  return response.data;
}

// Model Retraining
export async function retrainModel(
  request: RetrainingRequest
): Promise<RetrainingResult> {
  const response = await apiClient.post<RetrainingResult>('/model/retrain', request);
  return response.data;
}

// Error Handler
export function handleApiError(error: unknown): string {
  if (axios.isAxiosError(error)) {
    if (error.response) {
      // Server responded with error
      return error.response.data?.detail || error.message;
    } else if (error.request) {
      // Request made but no response
      return 'Unable to connect to the API server. Please ensure it is running.';
    }
  }
  return 'An unexpected error occurred';
}
