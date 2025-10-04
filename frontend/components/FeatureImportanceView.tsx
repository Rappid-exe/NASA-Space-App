'use client';

import { useEffect, useState } from 'react';
import { getFeatureImportance } from '@/lib/api';
import type { FeatureImportance } from '@/lib/types';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

export default function FeatureImportanceView() {
  const [importance, setImportance] = useState<FeatureImportance | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchImportance() {
      try {
        const data = await getFeatureImportance();
        setImportance(data);
      } catch (err) {
        setError('Failed to load feature importance. Ensure a model is loaded.');
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    fetchImportance();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-400">Loading feature importance...</p>
        </div>
      </div>
    );
  }

  if (error || !importance) {
    return (
      <div className="bg-yellow-900/20 border border-yellow-700 rounded-lg p-6 text-center">
        <p className="text-yellow-400">⚠️ {error || 'No feature importance data available'}</p>
      </div>
    );
  }

  const chartData = importance.features.slice(0, 15).map((f) => ({
    name: f.name.replace('koi_', '').replace('_', ' '),
    importance: f.importance * 100,
    fullName: f.name,
  }));

  const COLORS = ['#3b82f6', '#8b5cf6', '#ec4899', '#10b981', '#f59e0b'];

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-900/50 to-purple-900/50 backdrop-blur-sm rounded-lg p-6 border border-blue-800">
        <h2 className="text-2xl font-bold text-white mb-2">Feature Importance Analysis</h2>
        <p className="text-gray-300">
          Understanding which features contribute most to the model's predictions
        </p>
        <div className="mt-4 text-sm text-gray-400">
          <span className="font-semibold text-blue-400">Algorithm:</span> {importance.algorithm}
        </div>
      </div>

      {/* Top 5 Features */}
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <h3 className="text-xl font-semibold text-white mb-4">Top 5 Most Important Features</h3>
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          {importance.top_5.map((feature, index) => (
            <div
              key={feature.name}
              className="bg-gradient-to-br from-gray-700 to-gray-800 rounded-lg p-4 border border-gray-600"
            >
              <div className="text-3xl font-bold text-blue-400 mb-2">#{index + 1}</div>
              <div className="text-sm text-gray-300 mb-2 font-medium">
                {feature.name.replace('koi_', '').replace(/_/g, ' ')}
              </div>
              <div className="text-lg font-semibold text-white">
                {(feature.importance * 100).toFixed(2)}%
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Feature Importance Chart */}
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <h3 className="text-xl font-semibold text-white mb-4">Feature Importance Distribution</h3>
        <ResponsiveContainer width="100%" height={500}>
          <BarChart data={chartData} layout="vertical" margin={{ left: 120 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis type="number" stroke="#9ca3af" />
            <YAxis type="category" dataKey="name" stroke="#9ca3af" width={110} />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1f2937',
                border: '1px solid #374151',
                borderRadius: '8px',
                color: '#fff',
              }}
              formatter={(value: number) => [`${value.toFixed(2)}%`, 'Importance']}
            />
            <Bar dataKey="importance" radius={[0, 8, 8, 0]}>
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Feature Explanations */}
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <h3 className="text-xl font-semibold text-white mb-4">What Do These Features Mean?</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <FeatureExplanation
            name="Orbital Period"
            description="Time it takes for the planet to complete one orbit around its star"
            icon="🔄"
          />
          <FeatureExplanation
            name="Transit Duration"
            description="How long the planet takes to cross in front of its star"
            icon="⏱️"
          />
          <FeatureExplanation
            name="Transit Depth"
            description="How much the star's brightness decreases during transit"
            icon="📉"
          />
          <FeatureExplanation
            name="Planetary Radius"
            description="Size of the planet compared to Earth"
            icon="🌍"
          />
          <FeatureExplanation
            name="Equilibrium Temperature"
            description="Estimated temperature based on stellar radiation"
            icon="🌡️"
          />
          <FeatureExplanation
            name="Period-Duration Ratio"
            description="Derived feature comparing orbital period to transit duration"
            icon="📊"
          />
        </div>
      </div>

      {/* Interpretation Guide */}
      <div className="bg-blue-900/20 border border-blue-700 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-blue-400 mb-3">💡 How to Interpret</h3>
        <ul className="space-y-2 text-gray-300">
          <li className="flex items-start">
            <span className="text-blue-400 mr-2">•</span>
            <span>Higher importance values indicate features that have more influence on the model's predictions</span>
          </li>
          <li className="flex items-start">
            <span className="text-blue-400 mr-2">•</span>
            <span>Features with low importance may be redundant or not useful for distinguishing exoplanets from false positives</span>
          </li>
          <li className="flex items-start">
            <span className="text-blue-400 mr-2">•</span>
            <span>Different algorithms may assign different importance scores to the same features</span>
          </li>
        </ul>
      </div>
    </div>
  );
}

function FeatureExplanation({ name, description, icon }: { name: string; description: string; icon: string }) {
  return (
    <div className="bg-gray-700/50 rounded-lg p-4 border border-gray-600">
      <div className="flex items-center space-x-2 mb-2">
        <span className="text-2xl">{icon}</span>
        <h4 className="font-semibold text-white">{name}</h4>
      </div>
      <p className="text-sm text-gray-400">{description}</p>
    </div>
  );
}
