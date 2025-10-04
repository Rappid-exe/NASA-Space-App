'use client';

import { useState } from 'react';
import { retrainModel, handleApiError } from '@/lib/api';
import type { RetrainingRequest, RetrainingResult } from '@/lib/types';

export default function ModelRetraining() {
  const [algorithm, setAlgorithm] = useState('RandomForest');
  const [dataset, setDataset] = useState('kepler');
  const [customParams, setCustomParams] = useState('');
  const [useCustomParams, setUseCustomParams] = useState(false);
  const [retraining, setRetraining] = useState(false);
  const [result, setResult] = useState<RetrainingResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleRetrain = async () => {
    setRetraining(true);
    setError(null);
    setResult(null);

    try {
      let hyperparameters = undefined;
      
      if (useCustomParams && customParams.trim()) {
        try {
          hyperparameters = JSON.parse(customParams);
        } catch (e) {
          throw new Error('Invalid JSON in custom parameters');
        }
      }

      const request: RetrainingRequest = {
        algorithm,
        dataset,
        hyperparameters,
      };

      const data = await retrainModel(request);
      setResult(data);
    } catch (err) {
      setError(handleApiError(err));
    } finally {
      setRetraining(false);
    }
  };

  const defaultParams = {
    RandomForest: {
      n_estimators: 100,
      max_depth: null,
      min_samples_split: 2,
      random_state: 42,
    },
    NeuralNetwork: {
      hidden_layers: [128, 64, 32],
      dropout_rate: 0.3,
      learning_rate: 0.001,
      random_state: 42,
    },
    SVM: {
      kernel: 'rbf',
      C: 1.0,
      gamma: 'scale',
      random_state: 42,
    },
  };

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-900/50 to-purple-900/50 backdrop-blur-sm rounded-lg p-6 border border-blue-800">
        <h2 className="text-2xl font-bold text-white mb-2">Model Retraining</h2>
        <p className="text-gray-300">
          Train a new model with custom configuration and dataset selection
        </p>
      </div>

      {/* Configuration */}
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <h3 className="text-xl font-semibold text-white mb-4">Training Configuration</h3>
        
        <div className="space-y-4">
          {/* Algorithm Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Algorithm
            </label>
            <select
              value={algorithm}
              onChange={(e) => setAlgorithm(e.target.value)}
              className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={retraining}
            >
              <option value="RandomForest">Random Forest</option>
              <option value="NeuralNetwork">Neural Network</option>
              <option value="SVM">Support Vector Machine (SVM)</option>
            </select>
          </div>

          {/* Dataset Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Training Dataset
            </label>
            <select
              value={dataset}
              onChange={(e) => setDataset(e.target.value)}
              className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={retraining}
            >
              <option value="kepler">Kepler Mission</option>
              <option value="tess">TESS Mission</option>
              <option value="k2">K2 Mission</option>
            </select>
          </div>

          {/* Custom Parameters Toggle */}
          <div className="flex items-center space-x-2">
            <input
              type="checkbox"
              id="useCustomParams"
              checked={useCustomParams}
              onChange={(e) => setUseCustomParams(e.target.checked)}
              className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
              disabled={retraining}
            />
            <label htmlFor="useCustomParams" className="text-sm font-medium text-gray-300">
              Use Custom Hyperparameters
            </label>
          </div>

          {/* Default Parameters Display */}
          {!useCustomParams && (
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Default Parameters
              </label>
              <div className="bg-gray-900 rounded-lg p-4 border border-gray-700">
                <pre className="text-sm text-gray-300 overflow-x-auto">
                  {JSON.stringify(defaultParams[algorithm as keyof typeof defaultParams], null, 2)}
                </pre>
              </div>
            </div>
          )}

          {/* Custom Parameters Input */}
          {useCustomParams && (
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Custom Hyperparameters (JSON)
              </label>
              <textarea
                value={customParams}
                onChange={(e) => setCustomParams(e.target.value)}
                placeholder={JSON.stringify(defaultParams[algorithm as keyof typeof defaultParams], null, 2)}
                className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500 font-mono text-sm"
                rows={8}
                disabled={retraining}
              />
            </div>
          )}

          {/* Retrain Button */}
          <button
            onClick={handleRetrain}
            disabled={retraining}
            className="w-full px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg font-semibold hover:from-blue-700 hover:to-purple-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {retraining ? (
              <span className="flex items-center justify-center">
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
                Training... This may take several minutes
              </span>
            ) : (
              '🔄 Start Model Retraining'
            )}
          </button>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-900/20 border border-red-700 rounded-lg p-4">
          <p className="text-red-400">❌ {error}</p>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-4">
          {/* Success Message */}
          <div className="bg-green-900/20 border border-green-700 rounded-lg p-6">
            <h3 className="text-xl font-semibold text-green-400 mb-4">✅ Model Trained Successfully</h3>
            <div className="space-y-2 text-gray-300">
              <p><strong>Model ID:</strong> <span className="font-mono text-sm">{result.model_id}</span></p>
              <p><strong>Algorithm:</strong> {result.algorithm}</p>
              <p><strong>Dataset:</strong> {result.dataset}</p>
            </div>
          </div>

          {/* Performance Metrics */}
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
            <h3 className="text-xl font-semibold text-white mb-4">Performance Metrics</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <MetricCard
                label="Accuracy"
                value={result.performance.accuracy}
                color="blue"
              />
              <MetricCard
                label="Precision"
                value={result.performance.precision}
                color="purple"
              />
              <MetricCard
                label="Recall"
                value={result.performance.recall}
                color="pink"
              />
              <MetricCard
                label="F1 Score"
                value={result.performance.f1_score}
                color="green"
              />
            </div>
          </div>

          {/* Next Steps */}
          <div className="bg-blue-900/20 border border-blue-700 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-blue-400 mb-3">💡 Next Steps</h3>
            <p className="text-gray-300 mb-3">
              Your new model has been trained and automatically loaded for inference.
            </p>
            <ul className="space-y-2 text-gray-300 list-disc list-inside">
              <li>Test the model using the Classification page</li>
              <li>Upload a dataset to perform batch classification</li>
              <li>Compare performance with other models in the Dashboard</li>
            </ul>
          </div>
        </div>
      )}

      {/* Information */}
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-3">ℹ️ About Model Retraining</h3>
        <div className="space-y-2 text-gray-300 text-sm">
          <p>
            Model retraining allows you to create new models with different configurations and datasets.
          </p>
          <p>
            <strong>Datasets:</strong> Choose from Kepler, TESS, or K2 mission data. Each dataset has
            different characteristics and may produce models with varying performance.
          </p>
          <p>
            <strong>Hyperparameters:</strong> Use default parameters for quick training, or customize
            them based on hyperparameter tuning results for optimal performance.
          </p>
          <p>
            <strong>Note:</strong> Training time varies by algorithm and dataset size. Neural Networks
            typically take longer than Random Forest or SVM.
          </p>
        </div>
      </div>
    </div>
  );
}

function MetricCard({ label, value, color }: { label: string; value: number; color: string }) {
  const percentage = (value * 100).toFixed(2);
  
  const colorClasses = {
    blue: 'from-blue-900/50 to-blue-800/50 border-blue-700 text-blue-400',
    purple: 'from-purple-900/50 to-purple-800/50 border-purple-700 text-purple-400',
    pink: 'from-pink-900/50 to-pink-800/50 border-pink-700 text-pink-400',
    green: 'from-green-900/50 to-green-800/50 border-green-700 text-green-400',
  };

  return (
    <div className={`bg-gradient-to-br ${colorClasses[color as keyof typeof colorClasses]} backdrop-blur-sm rounded-lg p-4 border`}>
      <div className="text-sm text-gray-300 mb-1">{label}</div>
      <div className={`text-2xl font-bold ${colorClasses[color as keyof typeof colorClasses].split(' ')[2]}`}>
        {percentage}%
      </div>
    </div>
  );
}
