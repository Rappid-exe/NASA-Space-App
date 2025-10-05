'use client';

import { useState } from 'react';
import { classifyObservation, handleApiError } from '@/lib/api';
import type { ExoplanetFeatures, ClassificationResult } from '@/lib/types';

interface ClassificationFormProps {
  onResult: (result: ClassificationResult) => void;
  onLoadingChange: (loading: boolean) => void;
}

export default function ClassificationForm({ onResult, onLoadingChange }: ClassificationFormProps) {
  const [formData, setFormData] = useState<ExoplanetFeatures>({
    orbital_period: 3.52,
    transit_duration: 2.8,
    transit_depth: 500.0,
    planetary_radius: 1.2,
    equilibrium_temperature: 1200.0,
  });
  const [error, setError] = useState<string | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value === '' ? undefined : parseFloat(value),
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    onLoadingChange(true);

    try {
      // Validate inputs
      if (formData.orbital_period <= 0) {
        throw new Error('Orbital period must be positive');
      }
      if (formData.transit_duration <= 0) {
        throw new Error('Transit duration must be positive');
      }
      if (formData.transit_depth <= 0) {
        throw new Error('Transit depth must be positive');
      }
      if (formData.planetary_radius <= 0) {
        throw new Error('Planetary radius must be positive');
      }
      if (formData.equilibrium_temperature !== undefined && formData.equilibrium_temperature <= 0) {
        throw new Error('Equilibrium temperature must be positive');
      }

      const result = await classifyObservation(formData);
      onResult(result);
    } catch (err) {
      const errorMessage = handleApiError(err);
      setError(errorMessage);
    } finally {
      onLoadingChange(false);
    }
  };

  const loadExample = (type: 'hot-jupiter' | 'super-earth' | 'false-positive') => {
    const examples = {
      'hot-jupiter': {
        orbital_period: 3.52,
        transit_duration: 2.8,
        transit_depth: 15000.0,  // Deep transit for large planet
        planetary_radius: 11.2,   // Jupiter-sized (11.2 Earth radii)
        equilibrium_temperature: 1450.0,  // Very hot!
      },
      'super-earth': {
        orbital_period: 10.5,     // Shorter period than Earth
        transit_duration: 3.5,    // Moderate transit duration
        transit_depth: 450.0,     // Detectable signal
        planetary_radius: 2.5,    // Super-Earth size
        equilibrium_temperature: 650.0,  // Warm but not extreme
      },
      'false-positive': {
        orbital_period: 0.5,      // Unrealistically short
        transit_duration: 0.2,    // Too short
        transit_depth: 25000.0,   // Suspiciously deep
        planetary_radius: 0.3,    // Too small for such depth
        equilibrium_temperature: 3500.0,  // Extremely hot
      },
    };
    setFormData(examples[type]);
  };

  return (
    <div className="glass-strong rounded-2xl p-8 border border-white/30 shadow-xl hover:shadow-2xl transition-all duration-normal">
      <h3 className="font-display text-2xl font-semibold text-white mb-8 drop-shadow-lg">Observation Data</h3>

      {/* Quick Load Examples with refined styling */}
      <div className="mb-8">
        <label className="block text-base font-semibold text-gray-200 mb-3">
          Quick Load Example:
        </label>
        <div className="flex flex-wrap gap-3">
          <button
            type="button"
            onClick={() => loadExample('hot-jupiter')}
            className="px-4 py-2.5 bg-gradient-to-r from-orange-600 to-red-600 hover:from-orange-700 hover:to-red-700 text-white text-sm font-medium rounded-xl transition-all duration-fast transform hover:scale-105 hover:-translate-y-0.5 shadow-lg hover:shadow-glow-red"
          >
            🔥 Hot Jupiter
          </button>
          <button
            type="button"
            onClick={() => loadExample('super-earth')}
            className="px-4 py-2.5 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white text-sm font-medium rounded-xl transition-all duration-fast transform hover:scale-105 hover:-translate-y-0.5 shadow-lg hover:shadow-glow-green"
          >
            🌍 Super-Earth
          </button>
          <button
            type="button"
            onClick={() => loadExample('false-positive')}
            className="px-4 py-2.5 bg-gradient-to-r from-red-600 to-rose-600 hover:from-red-700 hover:to-rose-700 text-white text-sm font-medium rounded-xl transition-all duration-fast transform hover:scale-105 hover:-translate-y-0.5 shadow-lg hover:shadow-glow-red"
          >
            ❌ False Positive
          </button>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Orbital Period */}
        <div>
          <label htmlFor="orbital_period" className="block text-base font-semibold text-gray-200 mb-3">
            Orbital Period (days) *
          </label>
          <input
            type="number"
            id="orbital_period"
            name="orbital_period"
            value={formData.orbital_period}
            onChange={handleChange}
            step="0.01"
            required
            className="w-full px-5 py-3.5 bg-gray-900/80 border border-gray-700 rounded-xl text-white text-base focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-fast hover:border-gray-600"
            placeholder="e.g., 3.52"
          />
        </div>

        {/* Transit Duration */}
        <div>
          <label htmlFor="transit_duration" className="block text-base font-semibold text-gray-200 mb-3">
            Transit Duration (hours) *
          </label>
          <input
            type="number"
            id="transit_duration"
            name="transit_duration"
            value={formData.transit_duration}
            onChange={handleChange}
            step="0.01"
            required
            className="w-full px-5 py-3.5 bg-gray-900/80 border border-gray-700 rounded-xl text-white text-base focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-fast hover:border-gray-600"
            placeholder="e.g., 2.8"
          />
        </div>

        {/* Transit Depth */}
        <div>
          <label htmlFor="transit_depth" className="block text-base font-semibold text-gray-200 mb-3">
            Transit Depth (ppm) *
          </label>
          <input
            type="number"
            id="transit_depth"
            name="transit_depth"
            value={formData.transit_depth}
            onChange={handleChange}
            step="0.1"
            required
            className="w-full px-5 py-3.5 bg-gray-900/80 border border-gray-700 rounded-xl text-white text-base focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-fast hover:border-gray-600"
            placeholder="e.g., 500.0"
          />
        </div>

        {/* Planetary Radius */}
        <div>
          <label htmlFor="planetary_radius" className="block text-base font-semibold text-gray-200 mb-3">
            Planetary Radius (Earth radii) *
          </label>
          <input
            type="number"
            id="planetary_radius"
            name="planetary_radius"
            value={formData.planetary_radius}
            onChange={handleChange}
            step="0.01"
            required
            className="w-full px-5 py-3.5 bg-gray-900/80 border border-gray-700 rounded-xl text-white text-base focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-fast hover:border-gray-600"
            placeholder="e.g., 1.2"
          />
        </div>

        {/* Equilibrium Temperature */}
        <div>
          <label htmlFor="equilibrium_temperature" className="block text-base font-semibold text-gray-200 mb-3">
            Equilibrium Temperature (Kelvin)
            <span className="text-gray-400 ml-2 font-normal">(optional)</span>
          </label>
          <input
            type="number"
            id="equilibrium_temperature"
            name="equilibrium_temperature"
            value={formData.equilibrium_temperature || ''}
            onChange={handleChange}
            step="0.1"
            className="w-full px-5 py-3.5 bg-gray-900/80 border border-gray-700 rounded-xl text-white text-base focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-fast hover:border-gray-600"
            placeholder="e.g., 1200.0"
          />
        </div>

        {/* Error Message with refined styling */}
        {error && (
          <div className="bg-red-900/30 border border-red-600 rounded-xl p-5 animate-fadeInScale">
            <p className="text-red-300 text-base font-medium flex items-center gap-2">
              <span className="text-xl">❌</span>
              <span>{error}</span>
            </p>
          </div>
        )}

        {/* Submit Button with refined styling */}
        <button
          type="submit"
          className="w-full px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white text-lg font-bold rounded-2xl hover:from-blue-700 hover:to-purple-700 transition-all duration-normal transform hover:scale-105 hover:-translate-y-1 shadow-2xl hover:shadow-glow-blue border border-white/20 hover:border-white/40 mt-8"
        >
          🚀 Classify Observation
        </button>
      </form>
    </div>
  );
}
