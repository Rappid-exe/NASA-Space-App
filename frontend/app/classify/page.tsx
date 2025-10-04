'use client';

import { useState } from 'react';
import Link from 'next/link';
import ClassificationForm from '@/components/ClassificationForm';
import ResultsDisplay from '@/components/ResultsDisplay';
import type { ClassificationResult } from '@/lib/types';

export default function ClassifyPage() {
  const [result, setResult] = useState<ClassificationResult | null>(null);
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
                <p className="text-sm text-gray-400">Single Observation Classification</p>
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
              Classify Exoplanet Observation
            </h2>
            <p className="text-xl text-gray-300">
              Enter astronomical features to classify whether an observation is a confirmed exoplanet or false positive
            </p>
          </div>

          {/* Two Column Layout */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Left Column - Form */}
            <div className="space-y-6">
              <ClassificationForm
                onResult={setResult}
                onLoadingChange={setLoading}
              />

              {/* Info Card */}
              <div className="bg-blue-900/20 border border-blue-800 rounded-lg p-6">
                <h3 className="text-lg font-semibold text-blue-400 mb-3">ℹ️ About the Features</h3>
                <div className="space-y-2 text-sm text-gray-300">
                  <p><strong>Orbital Period:</strong> Time for planet to complete one orbit (days)</p>
                  <p><strong>Transit Duration:</strong> Time planet blocks star's light (hours)</p>
                  <p><strong>Transit Depth:</strong> Amount of light blocked (parts per million)</p>
                  <p><strong>Planetary Radius:</strong> Size relative to Earth (Earth radii)</p>
                  <p><strong>Equilibrium Temperature:</strong> Estimated surface temperature (Kelvin)</p>
                </div>
              </div>

              {/* Example Values */}
              <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
                <h3 className="text-lg font-semibold text-white mb-3">📝 Example Values</h3>
                <div className="space-y-3 text-sm">
                  <div>
                    <div className="text-gray-400 mb-1">Hot Jupiter (Confirmed):</div>
                    <div className="text-gray-300 font-mono text-xs">
                      Period: 3.5 days, Duration: 2.8 hrs, Depth: 500 ppm, Radius: 1.2 RE, Temp: 1200 K
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-400 mb-1">Earth-like (Confirmed):</div>
                    <div className="text-gray-300 font-mono text-xs">
                      Period: 365 days, Duration: 6.5 hrs, Depth: 84 ppm, Radius: 1.0 RE, Temp: 288 K
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-400 mb-1">False Positive:</div>
                    <div className="text-gray-300 font-mono text-xs">
                      Period: 0.5 days, Duration: 1.2 hrs, Depth: 2000 ppm, Radius: 0.8 RE, Temp: 2500 K
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Right Column - Results */}
            <div>
              {loading ? (
                <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-12 text-center">
                  <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                  <p className="text-gray-400">Classifying observation...</p>
                </div>
              ) : result ? (
                <ResultsDisplay result={result} />
              ) : (
                <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-12 text-center">
                  <div className="text-6xl mb-4">🔭</div>
                  <p className="text-gray-400 text-lg">
                    Enter observation data and click "Classify" to see results
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
