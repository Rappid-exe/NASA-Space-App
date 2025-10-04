'use client';

import { useState } from 'react';
import { tuneHyperparameters, handleApiError } from '@/lib/api';
import type { HyperparameterTuningRequest, HyperparameterTuningResult } from '@/lib/types';

export default function HyperparameterTuning() {
  const [algorithm, setAlgorithm] = useState('RandomForest');
  const [cvFolds, setCvFolds] = useState(5);
  const [tuning, setTuning] = useState(false);
  const [result, setResult] = useState<HyperparameterTuningResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Predefined parameter grids
  const paramGrids = {
    RandomForest: {
      n_estimators: [50, 100, 200],
      max_depth: [10, 20, 30, null],
      min_samples_split: [2, 5, 10],
    },
    SVM: {
      C: [0.1, 1, 10],
      kernel: ['rbf', 'linear'],
      gamma: ['scale', 'auto'],
    },
  };

  const handleTune = async () => {
    setTuning(true);
    setError(null);
    setResult(null);

    try {
      const request: HyperparameterTuningRequest = {
        algorithm,
        param_grid: paramGrids[algorithm as keyof typeof paramGrids],
        cv_folds: cvFolds,
      };

      const data = await tuneHyperparameters(request);
      setResult(data);
    } catch (err) {
      setError(handleApiError(err));
    } finally {
      setTuning(false);
    }
  };

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-900/50 to-purple-900/50 backdrop-blur-sm rounded-lg p-6 border border-blue-800">
        <h2 className="text-2xl font-bold text-white mb-2">Hyperparameter Tuning</h2>
        <p className="text-gray-300">
          Optimize model performance by finding the best hyperparameter configuration
        </p>
      </div>

      {/* Configuration */}
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <h3 className="text-xl font-semibold text-white mb-4">Tuning Configuration</h3>
        
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
              disabled={tuning}
            >
              <option value="RandomForest">Random Forest</option>
              <option value="SVM">Support Vector Machine (SVM)</option>
            </select>
          </div>

          {/* CV Folds */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Cross-Validation Folds: {cvFolds}
            </label>
            <input
              type="range"
              min="2"
              max="10"
              value={cvFolds}
              onChange={(e) => setCvFolds(parseInt(e.target.value))}
              className="w-full"
              disabled={tuning}
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>2</span>
              <span>10</span>
            </div>
          </div>

          {/* Parameter Grid Display */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Parameter Grid
            </label>
            <div className="bg-gray-900 rounded-lg p-4 border border-gray-700">
              <pre className="text-sm text-gray-300 overflow-x-auto">
                {JSON.stringify(paramGrids[algorithm as keyof typeof paramGrids], null, 2)}
              </pre>
            </div>
          </div>

          {/* Tune Button */}
          <button
            onClick={handleTune}
            disabled={tuning}
            className="w-full px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg font-semibold hover:from-blue-700 hover:to-purple-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {tuning ? (
              <span className="flex items-center justify-center">
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
                Tuning... This may take a few minutes
              </span>
            ) : (
              '⚙️ Start Hyperparameter Tuning'
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
          {/* Best Parameters */}
          <div className="bg-green-900/20 border border-green-700 rounded-lg p-6">
            <h3 className="text-xl font-semibold text-green-400 mb-4">✅ Best Parameters Found</h3>
            <div className="bg-gray-900 rounded-lg p-4 border border-gray-700">
              <pre className="text-sm text-gray-300 overflow-x-auto">
                {JSON.stringify(result.best_params, null, 2)}
              </pre>
            </div>
            <div className="mt-4">
              <div className="text-sm text-gray-400">Best Cross-Validation Score (F1)</div>
              <div className="text-3xl font-bold text-green-400 mt-1">
                {(result.best_score * 100).toFixed(2)}%
              </div>
            </div>
          </div>

          {/* CV Results */}
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
            <h3 className="text-xl font-semibold text-white mb-4">Cross-Validation Results</h3>
            <div className="space-y-2">
              {result.cv_results.mean_scores.map((score, index) => (
                <div key={index} className="flex items-center space-x-4">
                  <span className="text-gray-400 w-24">Config {index + 1}</span>
                  <div className="flex-1 bg-gray-700 rounded-full h-6 relative">
                    <div
                      className="bg-gradient-to-r from-blue-500 to-purple-500 h-6 rounded-full flex items-center justify-end pr-2"
                      style={{ width: `${score * 100}%` }}
                    >
                      <span className="text-xs text-white font-semibold">
                        {(score * 100).toFixed(2)}%
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Usage Instructions */}
          <div className="bg-blue-900/20 border border-blue-700 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-blue-400 mb-3">💡 Next Steps</h3>
            <p className="text-gray-300 mb-3">
              Use these optimized parameters to retrain your model for better performance:
            </p>
            <ol className="space-y-2 text-gray-300 list-decimal list-inside">
              <li>Go to the "Model Retraining" tab</li>
              <li>Select the same algorithm ({algorithm})</li>
              <li>Copy the best parameters shown above</li>
              <li>Start retraining with the optimized configuration</li>
            </ol>
          </div>
        </div>
      )}

      {/* Information */}
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-3">ℹ️ About Hyperparameter Tuning</h3>
        <div className="space-y-2 text-gray-300 text-sm">
          <p>
            Hyperparameter tuning uses grid search with cross-validation to find the best parameter
            combination for your model.
          </p>
          <p>
            <strong>Cross-Validation:</strong> The dataset is split into {cvFolds} folds, and the model
            is trained and evaluated {cvFolds} times to ensure robust performance estimates.
          </p>
          <p>
            <strong>Note:</strong> This process can take several minutes depending on the parameter grid
            size and dataset complexity.
          </p>
        </div>
      </div>
    </div>
  );
}
