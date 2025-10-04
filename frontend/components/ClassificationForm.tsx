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

  const loadExample = (type: 'hot-jupiter' | 'earth-like' | 'false-positive') => {
    const examples = {
      'hot-jupiter': {
        orbital_period: 3.52,
        transit_duration: 2.8,
        transit_depth: 500.0,
        planetary_radius: 1.2,
        equilibrium_temperature: 1200.0,
      },
      'earth-like': {
        orbital_period: 365.25,
        transit_duration: 6.5,
        transit_depth: 84.0,
        planetary_radius: 1.0,
        equilibrium_temperature: 288.0,
      },
      'false-positive': {
        orbital_period: 0.5,
        transit_duration: 1.2,
        transit_depth: 2000.0,
        planetary_radius: 0.8,
        equilibrium_temperature: 2500.0,
      },
    };
    setFormData(examples[type]);
  };

  return (
    <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
      <h3 className="text-xl font-semibold text-white mb-6">Observation Data</h3>

      {/* Quick Load Examples */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Quick Load Example:
        </label>
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => loadExample('hot-jupiter')}
            className="px-3 py-1 bg-orange-600 hover:bg-orange-700 text-white text-sm rounded transition-colors"
          >
            🔥 Hot Jupiter
          </button>
          <button
            type="button"
            onClick={() => loadExample('earth-like')}
            className="px-3 py-1 bg-green-600 hover:bg-green-700 text-white text-sm rounded transition-colors"
          >
            🌍 Earth-like
          </button>
          <button
            type="button"
            onClick={() => loadExample('false-positive')}
            className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-sm rounded transition-colors"
          >
            ❌ False Positive
          </button>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Orbital Period */}
        <div>
          <label htmlFor="orbital_period" className="block text-sm font-medium text-gray-300 mb-2">
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
            className="w-full px-4 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            placeholder="e.g., 3.52"
          />
        </div>

        {/* Transit Duration */}
        <div>
          <label htmlFor="transit_duration" className="block text-sm font-medium text-gray-300 mb-2">
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
            className="w-full px-4 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            placeholder="e.g., 2.8"
          />
        </div>

        {/* Transit Depth */}
        <div>
          <label htmlFor="transit_depth" className="block text-sm font-medium text-gray-300 mb-2">
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
            className="w-full px-4 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            placeholder="e.g., 500.0"
          />
        </div>

        {/* Planetary Radius */}
        <div>
          <label htmlFor="planetary_radius" className="block text-sm font-medium text-gray-300 mb-2">
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
            className="w-full px-4 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            placeholder="e.g., 1.2"
          />
        </div>

        {/* Equilibrium Temperature */}
        <div>
          <label htmlFor="equilibrium_temperature" className="block text-sm font-medium text-gray-300 mb-2">
            Equilibrium Temperature (Kelvin)
            <span className="text-gray-500 ml-2">(optional)</span>
          </label>
          <input
            type="number"
            id="equilibrium_temperature"
            name="equilibrium_temperature"
            value={formData.equilibrium_temperature || ''}
            onChange={handleChange}
            step="0.1"
            className="w-full px-4 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            placeholder="e.g., 1200.0"
          />
        </div>

        {/* Error Message */}
        {error && (
          <div className="bg-red-900/20 border border-red-700 rounded-lg p-4">
            <p className="text-red-400 text-sm">❌ {error}</p>
          </div>
        )}

        {/* Submit Button */}
        <button
          type="submit"
          className="w-full px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all transform hover:scale-105 shadow-lg hover:shadow-blue-500/50"
        >
          🚀 Classify Observation
        </button>
      </form>
    </div>
  );
}
