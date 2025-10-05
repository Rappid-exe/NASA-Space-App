'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import Dashboard from '@/components/Dashboard';
import { checkHealth } from '@/lib/api';
import type { HealthResponse } from '@/lib/types';

export default function DashboardPage() {
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
              <Link href="/" className="flex items-center space-x-3 hover:opacity-80 transition-opacity">
                <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                  <span className="text-2xl">🪐</span>
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-white">Exoplanet Classifier</h1>
                  <p className="text-sm text-gray-400">Dashboard</p>
                </div>
              </Link>
            </div>
            
            {/* Navigation */}
            <nav className="hidden md:flex items-center space-x-6">
              <Link 
                href="/"
                className="text-gray-400 hover:text-white transition-colors"
              >
                Home
              </Link>
              <Link 
                href="/classify"
                className="text-gray-400 hover:text-white transition-colors"
              >
                Classify
              </Link>
              <Link 
                href="/upload"
                className="text-gray-400 hover:text-white transition-colors"
              >
                Upload
              </Link>
              <Link 
                href="/advanced"
                className="text-gray-400 hover:text-white transition-colors"
              >
                Advanced
              </Link>
            </nav>

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

      {/* Dashboard Content with refined styling */}
      <section className="container mx-auto px-6 py-16 page-transition">
        <div className="mb-12 space-y-4 animate-fadeInUp">
          <h2 className="font-display text-5xl font-bold text-white drop-shadow-lg">Model Dashboard</h2>
          <p className="text-xl text-gray-300 leading-relaxed">View model statistics, performance metrics, and system information</p>
        </div>
        <div className="animate-fadeInScale">
          <Dashboard />
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
