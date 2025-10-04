'use client';

import { useState } from 'react';
import Link from 'next/link';
import FileUpload from '@/components/FileUpload';
import type { BatchClassificationResponse } from '@/lib/types';

export default function UploadPage() {
  const [results, setResults] = useState<BatchClassificationResponse | null>(null);
  const [loading, setLoading] = useState(false);

  return (
    <div className="min-h-screen bg-gradient-to-b from-space-darker via-space-dark to-space-darker">
      {/* Header */}
      <header className="border-b border-gray-800 bg-space-darker/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <Link href="/" className="flex items-center space-x-3 hover:opacity-80 transition-opacity">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                <span className="text-2xl">🪐</span>
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">Exoplanet Classifier</h1>
                <p className="text-sm text-gray-400">Batch Classification</p>
              </div>
            </Link>
            
            <Link
              href="/"
              className="px-4 py-2 bg-gray-800 text-white rounded-lg hover:bg-gray-700 transition-colors border border-gray-700"
            >
              ← Back to Home
            </Link>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-12">
        <div className="max-w-6xl mx-auto">
          {/* Page Title */}
          <div className="text-center mb-12 animate-fadeIn">
            <h2 className="text-4xl font-bold text-white mb-4">
              Batch Classification
            </h2>
            <p className="text-xl text-gray-300">
              Upload a CSV file to classify multiple exoplanet observations at once
            </p>
          </div>

          {/* Upload Section */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Left Column - Upload */}
            <div className="space-y-6">
              <FileUpload
                onResults={setResults}
                onLoadingChange={setLoading}
              />

              {/* CSV Format Info */}
              <div className="bg-blue-900/20 border border-blue-800 rounded-lg p-6">
                <h3 className="text-lg font-semibold text-blue-400 mb-3">📋 CSV Format</h3>
                <p className="text-sm text-gray-300 mb-3">
                  Your CSV file should have the following columns:
                </p>
                <div className="bg-gray-900 rounded p-3 font-mono text-xs text-gray-300 overflow-x-auto">
                  orbital_period,transit_duration,transit_depth,planetary_radius,equilibrium_temperature
                </div>
                <p className="text-sm text-gray-400 mt-3">
                  Note: equilibrium_temperature is optional
                </p>
              </div>

              {/* Download Sample */}
              <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
                <h3 className="text-lg font-semibold text-white mb-3">📥 Sample CSV</h3>
                <p className="text-sm text-gray-300 mb-4">
                  Download a sample CSV file to see the expected format
                </p>
                <button
                  onClick={() => {
                    const csv = `orbital_period,transit_duration,transit_depth,planetary_radius,equilibrium_temperature
3.52,2.8,500.0,1.2,1200.0
365.25,6.5,84.0,1.0,288.0
0.5,1.2,2000.0,0.8,2500.0`;
                    const blob = new Blob([csv], { type: 'text/csv' });
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'sample_exoplanets.csv';
                    a.click();
                  }}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
                >
                  Download Sample CSV
                </button>
              </div>
            </div>

            {/* Right Column - Results */}
            <div>
              {loading ? (
                <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-12 text-center">
                  <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                  <p className="text-gray-400">Processing observations...</p>
                </div>
              ) : results ? (
                <BatchResults results={results} />
              ) : (
                <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-12 text-center">
                  <div className="text-6xl mb-4">📊</div>
                  <p className="text-gray-400 text-lg">
                    Upload a CSV file to see batch classification results
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function BatchResults({ results }: { results: BatchClassificationResponse }) {
  const confirmedPercentage = ((results.summary.CONFIRMED / results.total_processed) * 100).toFixed(1);
  const falsePositivePercentage = ((results.summary.FALSE_POSITIVE / results.total_processed) * 100).toFixed(1);

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Summary Card */}
      <div className="bg-gradient-to-br from-blue-900/50 to-purple-900/50 backdrop-blur-sm rounded-lg p-6 border border-blue-800">
        <h3 className="text-2xl font-bold text-white mb-4">Batch Results Summary</h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-gray-900/50 rounded-lg p-4">
            <div className="text-3xl font-bold text-blue-400 mb-1">{results.total_processed}</div>
            <div className="text-gray-300 text-sm">Total Processed</div>
          </div>
          <div className="bg-gray-900/50 rounded-lg p-4">
            <div className="text-3xl font-bold text-green-400 mb-1">{results.summary.CONFIRMED}</div>
            <div className="text-gray-300 text-sm">Confirmed ({confirmedPercentage}%)</div>
          </div>
        </div>
      </div>

      {/* Distribution */}
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <h4 className="text-lg font-semibold text-white mb-4">Classification Distribution</h4>
        <div className="space-y-4">
          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-gray-300">✅ Confirmed Exoplanets</span>
              <span className="text-green-400 font-semibold">{results.summary.CONFIRMED} ({confirmedPercentage}%)</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-4">
              <div
                className="bg-green-500 h-4 rounded-full transition-all duration-500"
                style={{ width: `${confirmedPercentage}%` }}
              ></div>
            </div>
          </div>
          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-gray-300">❌ False Positives</span>
              <span className="text-red-400 font-semibold">{results.summary.FALSE_POSITIVE} ({falsePositivePercentage}%)</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-4">
              <div
                className="bg-red-500 h-4 rounded-full transition-all duration-500"
                style={{ width: `${falsePositivePercentage}%` }}
              ></div>
            </div>
          </div>
        </div>
      </div>

      {/* Individual Results */}
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <h4 className="text-lg font-semibold text-white mb-4">Individual Results</h4>
        <div className="max-h-96 overflow-y-auto space-y-2">
          {results.results.map((result, index) => (
            <div
              key={index}
              className={`p-3 rounded-lg border ${
                result.prediction === 'CONFIRMED'
                  ? 'bg-green-900/20 border-green-700'
                  : 'bg-red-900/20 border-red-700'
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <span className="text-2xl">{result.prediction === 'CONFIRMED' ? '✅' : '❌'}</span>
                  <div>
                    <div className="text-white font-semibold">Observation #{index + 1}</div>
                    <div className="text-sm text-gray-400">{result.prediction}</div>
                  </div>
                </div>
                <div className="text-right">
                  <div className={`text-lg font-bold ${
                    result.prediction === 'CONFIRMED' ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {(result.confidence * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-gray-400">confidence</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Export Button */}
      <button
        onClick={() => {
          const csv = ['observation,prediction,confidence,confirmed_probability,false_positive_probability'];
          results.results.forEach((result, index) => {
            csv.push(
              `${index + 1},${result.prediction},${result.confidence},${result.probabilities.CONFIRMED},${result.probabilities.FALSE_POSITIVE}`
            );
          });
          const blob = new Blob([csv.join('\n')], { type: 'text/csv' });
          const url = window.URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = 'classification_results.csv';
          a.click();
        }}
        className="w-full px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all"
      >
        📥 Export Results as CSV
      </button>
    </div>
  );
}
