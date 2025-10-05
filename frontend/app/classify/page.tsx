'use client';

import { useState, Suspense } from 'react';
import Link from 'next/link';
import dynamic from 'next/dynamic';
import ClassificationForm from '@/components/ClassificationForm';
import ResultsDisplay from '@/components/ResultsDisplay';
import LoadingScreen from '@/components/3d/effects/LoadingScreen';
import type { ClassificationResult } from '@/lib/types';

// Dynamically import 3D scene to avoid SSR issues
const ClassificationScene = dynamic(
  () => import('@/components/3d/scenes/ClassificationScene'),
  { ssr: false }
);

export default function ClassifyPage() {
  const [result, setResult] = useState<ClassificationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [sceneLoaded, setSceneLoaded] = useState(false);

  return (
    <div className="relative min-h-screen w-full overflow-hidden">
      {/* Loading Screen */}
      {!sceneLoaded && <LoadingScreen onLoadComplete={() => setSceneLoaded(true)} />}

      {/* 3D Classification Planet Background - Full Screen */}
      <div className="fixed inset-0 z-0 pointer-events-none">
        <Suspense fallback={
          <div className="w-full h-full bg-gradient-to-b from-space-darker via-space-dark to-space-darker" />
        }>
          <ClassificationScene 
            result={result?.prediction || null}
            isClassifying={loading}
            showParticles={true}
          />
        </Suspense>
      </div>

      {/* Header */}
      <header className="relative z-50 border-b border-white/10 bg-black/20 backdrop-blur-md sticky top-0">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <Link href="/" className="flex items-center space-x-3 hover:opacity-80 transition-opacity">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center shadow-lg">
                <span className="text-2xl">🪐</span>
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white drop-shadow-lg">Exoplanet Classifier</h1>
                <p className="text-sm text-gray-300 drop-shadow">Single Observation Classification</p>
              </div>
            </Link>
            
            <Link
              href="/"
              className="px-4 py-2 bg-black/40 backdrop-blur-md text-white rounded-lg hover:bg-black/60 transition-colors border border-white/30 hover:border-white/50 shadow-xl"
            >
              ← Back to Home
            </Link>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="relative z-10 container mx-auto px-6 py-16 page-transition">
        <div className="max-w-7xl mx-auto">
          {/* Page Title with refined typography */}
          <div className="text-center mb-16 space-y-6 animate-fadeInUp">
            <h2 className="font-display text-5xl md:text-6xl font-bold text-white drop-shadow-2xl leading-tight">
              Classify Exoplanet Observation
            </h2>
            <p className="text-xl md:text-2xl text-gray-100 drop-shadow-lg max-w-3xl mx-auto glass-strong rounded-2xl px-8 py-6 border border-white/20 leading-relaxed">
              Enter astronomical features to classify whether an observation is a confirmed exoplanet or false positive
            </p>
          </div>

          {/* Two Column Layout with refined spacing */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 animate-fadeInScale">
            {/* Left Column - Form */}
            <div className="space-y-6">
              <ClassificationForm
                onResult={setResult}
                onLoadingChange={setLoading}
              />

              {/* Info Card with refined styling */}
              <div className="glass-strong border border-blue-500/40 rounded-2xl p-7 shadow-xl hover:shadow-glow-blue transition-all duration-normal card-hover">
                <h3 className="font-display text-xl font-semibold text-blue-300 mb-4 drop-shadow flex items-center gap-2">
                  <span className="text-2xl">ℹ️</span>
                  <span>About the Features</span>
                </h3>
                <div className="space-y-3 text-base text-gray-200 drop-shadow leading-relaxed">
                  <p><strong className="text-blue-200">Orbital Period:</strong> Time for planet to complete one orbit (days)</p>
                  <p><strong className="text-blue-200">Transit Duration:</strong> Time planet blocks star's light (hours)</p>
                  <p><strong className="text-blue-200">Transit Depth:</strong> Amount of light blocked (parts per million)</p>
                  <p><strong className="text-blue-200">Planetary Radius:</strong> Size relative to Earth (Earth radii)</p>
                  <p><strong className="text-blue-200">Equilibrium Temperature:</strong> Estimated surface temperature (Kelvin)</p>
                </div>
              </div>

              {/* Example Values with refined styling */}
              <div className="glass-strong border border-white/30 rounded-2xl p-7 shadow-xl hover:shadow-2xl transition-all duration-normal card-hover">
                <h3 className="font-display text-xl font-semibold text-white mb-4 drop-shadow flex items-center gap-2">
                  <span className="text-2xl">📝</span>
                  <span>Example Values</span>
                </h3>
                <div className="space-y-4 text-sm">
                  <div className="p-3 bg-white/5 rounded-lg border border-white/10 hover:border-white/20 transition-colors">
                    <div className="text-gray-200 mb-2 font-semibold drop-shadow">Hot Jupiter (Confirmed):</div>
                    <div className="text-gray-300 font-mono text-xs drop-shadow leading-relaxed">
                      Period: 3.5 days, Duration: 2.8 hrs, Depth: 500 ppm, Radius: 1.2 RE, Temp: 1200 K
                    </div>
                  </div>
                  <div className="p-3 bg-white/5 rounded-lg border border-white/10 hover:border-white/20 transition-colors">
                    <div className="text-gray-200 mb-2 font-semibold drop-shadow">Super-Earth (Confirmed):</div>
                    <div className="text-gray-300 font-mono text-xs drop-shadow leading-relaxed">
                      Period: 10.5 days, Duration: 3.5 hrs, Depth: 450 ppm, Radius: 2.5 RE, Temp: 650 K
                    </div>
                  </div>
                  <div className="p-3 bg-white/5 rounded-lg border border-white/10 hover:border-white/20 transition-colors">
                    <div className="text-gray-200 mb-2 font-semibold drop-shadow">False Positive:</div>
                    <div className="text-gray-300 font-mono text-xs drop-shadow leading-relaxed">
                      Period: 0.5 days, Duration: 1.2 hrs, Depth: 2000 ppm, Radius: 0.8 RE, Temp: 2500 K
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Right Column - Results with refined styling */}
            <div>
              {loading ? (
                <div className="glass-strong border border-white/30 rounded-2xl p-16 text-center shadow-xl animate-fadeInScale">
                  <div className="w-20 h-20 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-6"></div>
                  <p className="text-gray-100 text-xl drop-shadow font-medium">Classifying observation...</p>
                </div>
              ) : result ? (
                <ResultsDisplay result={result} />
              ) : (
                <div className="glass-strong border border-white/30 rounded-2xl p-16 text-center shadow-xl hover:shadow-2xl transition-all duration-normal">
                  <div className="text-7xl mb-6 animate-pulse-slow">🔭</div>
                  <p className="text-gray-100 text-xl drop-shadow leading-relaxed">
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
