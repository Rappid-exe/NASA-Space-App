'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import Dashboard from '@/components/Dashboard';
import { checkHealth } from '@/lib/api';
import type { HealthResponse } from '@/lib/types';

export default function Home() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchHealth() {
      try {
        const data = await checkHealth();
        setHealth(data);
      } catch (error) {
        console.error('Failed to fetch health:', error);
      } finally {
        setLoading(false);
      }
    }
    fetchHealth();
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-b from-space-darker via-space-dark to-space-darker">
      {/* Header */}
      <header className="border-b border-gray-800 bg-space-darker/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                <span className="text-2xl">🪐</span>
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">Exoplanet Classifier</h1>
                <p className="text-sm text-gray-400">AI-Powered Discovery</p>
              </div>
            </div>
            
            {/* Status Indicator */}
            <div className="flex items-center space-x-2">
              {loading ? (
                <div className="flex items-center space-x-2 text-gray-400">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse"></div>
                  <span className="text-sm">Connecting...</span>
                </div>
              ) : health?.model_loaded ? (
                <div className="flex items-center space-x-2 text-green-400">
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse-slow"></div>
                  <span className="text-sm">Model Ready</span>
                </div>
              ) : (
                <div className="flex items-center space-x-2 text-yellow-400">
                  <div className="w-2 h-2 bg-yellow-400 rounded-full"></div>
                  <span className="text-sm">No Model Loaded</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="container mx-auto px-4 py-16 text-center">
        <div className="max-w-4xl mx-auto animate-fadeIn">
          <h2 className="text-5xl md:text-6xl font-bold text-white mb-6 bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
            Discover Exoplanets with AI
          </h2>
          <p className="text-xl text-gray-300 mb-8">
            Harness the power of machine learning trained on NASA's Kepler, TESS, and K2 missions 
            to identify and classify exoplanet candidates with unprecedented accuracy.
          </p>
          
          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center mb-12">
            <Link 
              href="/classify"
              className="px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg font-semibold hover:from-blue-700 hover:to-purple-700 transition-all transform hover:scale-105 shadow-lg hover:shadow-blue-500/50"
            >
              🚀 Start Classifying
            </Link>
            <Link 
              href="/upload"
              className="px-8 py-4 bg-gray-800 text-white rounded-lg font-semibold hover:bg-gray-700 transition-all border border-gray-700 hover:border-gray-600"
            >
              📊 Upload Dataset
            </Link>
            <Link 
              href="/advanced"
              className="px-8 py-4 bg-gray-800 text-white rounded-lg font-semibold hover:bg-gray-700 transition-all border border-gray-700 hover:border-gray-600"
            >
              ⚙️ Advanced Features
            </Link>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-3xl mx-auto">
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
              <div className="text-3xl font-bold text-blue-400 mb-2">3</div>
              <div className="text-gray-300">NASA Missions</div>
              <div className="text-sm text-gray-500 mt-1">Kepler, TESS, K2</div>
            </div>
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
              <div className="text-3xl font-bold text-purple-400 mb-2">10K+</div>
              <div className="text-gray-300">Training Samples</div>
              <div className="text-sm text-gray-500 mt-1">Validated exoplanets</div>
            </div>
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
              <div className="text-3xl font-bold text-pink-400 mb-2">92%+</div>
              <div className="text-gray-300">Accuracy</div>
              <div className="text-sm text-gray-500 mt-1">Model performance</div>
            </div>
          </div>
        </div>
      </section>

      {/* Dashboard Section */}
      <section className="container mx-auto px-4 py-12">
        <Dashboard />
      </section>

      {/* Features Section */}
      <section className="container mx-auto px-4 py-16">
        <h3 className="text-3xl font-bold text-white text-center mb-12">
          Powerful Features
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 max-w-6xl mx-auto">
          <FeatureCard
            icon="🎯"
            title="Real-Time Classification"
            description="Instantly classify exoplanet candidates with confidence scores and detailed explanations"
          />
          <FeatureCard
            icon="📈"
            title="Batch Processing"
            description="Upload CSV files to classify thousands of observations in seconds"
          />
          <FeatureCard
            icon="🧠"
            title="Multiple Algorithms"
            description="Choose from Random Forest, Neural Networks, and SVM classifiers"
          />
          <FeatureCard
            icon="📊"
            title="Performance Metrics"
            description="View detailed accuracy, precision, recall, and F1 scores for each model"
          />
          <FeatureCard
            icon="🔍"
            title="Feature Analysis"
            description="Understand which astronomical features contribute most to predictions"
          />
          <FeatureCard
            icon="🌌"
            title="NASA Data"
            description="Trained on authentic data from Kepler, TESS, and K2 space missions"
          />
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-gray-800 mt-16">
        <div className="container mx-auto px-4 py-8">
          <div className="text-center text-gray-400">
            <p className="mb-2">Built with Next.js, FastAPI, and TensorFlow</p>
            <p className="text-sm">Data from NASA Exoplanet Archive</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

function FeatureCard({ icon, title, description }: { icon: string; title: string; description: string }) {
  return (
    <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700 card-hover">
      <div className="text-4xl mb-4">{icon}</div>
      <h4 className="text-xl font-semibold text-white mb-2">{title}</h4>
      <p className="text-gray-400">{description}</p>
    </div>
  );
}
