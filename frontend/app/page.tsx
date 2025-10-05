'use client';

import { useEffect, useState, Suspense } from 'react';
import Link from 'next/link';
import dynamic from 'next/dynamic';
import { checkHealth } from '@/lib/api';
import type { HealthResponse } from '@/lib/types';
import LoadingScreen from '@/components/3d/effects/LoadingScreen';

// Dynamically import 3D scene to avoid SSR issues
const SolarSystemScene = dynamic(
  () => import('@/components/3d/scenes/SolarSystemScene'),
  { ssr: false }
);

export default function Home() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [sceneLoaded, setSceneLoaded] = useState(false);

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
    <div className="relative min-h-screen w-full overflow-hidden">
      {/* Loading Screen */}
      {!sceneLoaded && <LoadingScreen onLoadComplete={() => setSceneLoaded(true)} />}

      {/* 3D Solar System Background - Full Screen */}
      <div className="fixed inset-0 z-0 pointer-events-none">
        <Suspense fallback={
          <div className="w-full h-full bg-gradient-to-b from-space-darker via-space-dark to-space-darker" />
        }>
          <SolarSystemScene />
        </Suspense>
      </div>

      {/* Transparent Header Overlay */}
      <header className="relative z-50 border-b border-white/10 bg-black/20 backdrop-blur-md">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center shadow-lg">
                <span className="text-2xl">🪐</span>
              </div>
              <div>
                <h1 className="text-2xl font-nasa font-bold text-white drop-shadow-lg tracking-wide">EXOPLANET CLASSIFIER</h1>
                <p className="text-sm text-gray-300 drop-shadow">AI-Powered Discovery</p>
              </div>
            </div>
            
            {/* Navigation Links */}
            <nav className="hidden md:flex items-center space-x-6">
              <Link 
                href="/classify"
                className="text-white/90 hover:text-white transition-colors drop-shadow"
              >
                Classify
              </Link>
              <Link 
                href="/upload"
                className="text-white/90 hover:text-white transition-colors drop-shadow"
              >
                Upload
              </Link>
              <Link 
                href="/advanced"
                className="text-white/90 hover:text-white transition-colors drop-shadow"
              >
                Advanced
              </Link>
              <Link 
                href="/dashboard"
                className="text-white/90 hover:text-white transition-colors drop-shadow"
              >
                Dashboard
              </Link>
            </nav>

            {/* Status Indicator */}
            <div className="flex items-center space-x-2">
              {loading ? (
                <div className="flex items-center space-x-2 text-gray-300">
                  <div className="w-2 h-2 bg-gray-300 rounded-full animate-pulse"></div>
                  <span className="text-sm drop-shadow">Connecting...</span>
                </div>
              ) : health?.model_loaded ? (
                <div className="flex items-center space-x-2 text-green-300">
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse-slow shadow-lg shadow-green-400/50"></div>
                  <span className="text-sm drop-shadow">Ready</span>
                </div>
              ) : (
                <div className="flex items-center space-x-2 text-yellow-300">
                  <div className="w-2 h-2 bg-yellow-400 rounded-full"></div>
                  <span className="text-sm drop-shadow">Offline</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Hero Section - Centered Content Overlay */}
      <main className="relative z-10 flex items-center justify-center min-h-[calc(100vh-80px)] page-transition">
        <div className="container mx-auto px-6 text-center">
          <div className="max-w-5xl mx-auto space-y-8">
            {/* Main Title with NASA font */}
            <h2 className="font-nasa text-6xl md:text-8xl font-extrabold text-white mb-8 drop-shadow-2xl animate-fadeInUp leading-tight tracking-wider">
              <span className="gradient-text bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent inline-block">
                DISCOVER EXOPLANETS
              </span>
              <br />
              <span className="text-white inline-block mt-2">WITH AI</span>
            </h2>
            
            {/* Tagline with refined spacing and readability */}
            <p className="text-xl md:text-2xl text-gray-100 mb-12 drop-shadow-lg max-w-3xl mx-auto glass-strong rounded-2xl px-8 py-6 border border-white/20 leading-relaxed animate-fadeInScale">
              Harness machine learning trained on NASA's Kepler, TESS, and K2 missions 
              to classify exoplanet candidates.
            </p>
            
            {/* CTA Buttons with refined styling and animations */}
            <div className="flex flex-col sm:flex-row gap-5 justify-center items-center animate-fadeIn" style={{ animationDelay: '200ms' }}>
              <Link 
                href="/classify"
                className="group px-12 py-5 bg-gradient-to-r from-blue-600 to-purple-600 text-white text-lg rounded-2xl font-semibold hover:from-blue-700 hover:to-purple-700 transition-all duration-normal transform hover:scale-105 hover:-translate-y-1 shadow-2xl hover:shadow-glow-blue border border-white/30 hover:border-white/50 min-w-[240px]"
              >
                <span className="flex items-center justify-center gap-2">
                  <span className="text-2xl group-hover:scale-110 transition-transform">🚀</span>
                  <span>Start Classifying</span>
                </span>
              </Link>
              <Link 
                href="/dashboard"
                className="group px-12 py-5 glass-strong text-white text-lg rounded-2xl font-semibold hover:bg-black/70 transition-all duration-normal border border-white/30 hover:border-white/60 shadow-xl hover:shadow-2xl transform hover:scale-105 hover:-translate-y-1 min-w-[240px]"
              >
                <span className="flex items-center justify-center gap-2">
                  <span className="text-2xl group-hover:scale-110 transition-transform">📊</span>
                  <span>View Dashboard</span>
                </span>
              </Link>
            </div>
          </div>
        </div>
      </main>

      {/* Subtle gradient overlay at bottom for depth */}
      <div className="fixed bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-black/50 to-transparent pointer-events-none z-5" />
    </div>
  );
}
