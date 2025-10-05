'use client';

import { useEffect, useState } from 'react';
import { getModelStatistics } from '@/lib/api';
import type { ModelStatistics } from '@/lib/types';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

export default function Dashboard() {
  const [stats, setStats] = useState<ModelStatistics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchStats() {
      try {
        const data = await getModelStatistics();
        setStats(data);
      } catch (err) {
        setError('Failed to load model statistics. Please ensure the API is running and a model is loaded.');
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    fetchStats();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-400">Loading model statistics...</p>
        </div>
      </div>
    );
  }

  if (error || !stats) {
    return (
      <div className="bg-yellow-900/20 border border-yellow-700 rounded-lg p-6 text-center">
        <p className="text-yellow-400 mb-2">⚠️ {error || 'No model statistics available'}</p>
        <p className="text-sm text-gray-400">Train a model first by running: python test_model_training.py</p>
      </div>
    );
  }

  const metricsData = [
    { name: 'Accuracy', value: stats.accuracy * 100 },
    { name: 'Precision', value: stats.precision * 100 },
    { name: 'Recall', value: stats.recall * 100 },
    { name: 'F1 Score', value: stats.f1_score * 100 },
  ];

  const COLORS = ['#3b82f6', '#8b5cf6', '#ec4899', '#10b981'];

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Model Info Header */}
      <div className="bg-gradient-to-r from-blue-900/50 to-purple-900/50 backdrop-blur-sm rounded-lg p-6 border border-blue-800">
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div>
            <h3 className="text-2xl font-bold text-white mb-2">Current Model</h3>
            <div className="flex items-center space-x-4 text-sm">
              <span className="text-gray-300">
                <span className="text-gray-500">Algorithm:</span> <span className="font-semibold text-blue-400">{stats.algorithm}</span>
              </span>
              <span className="text-gray-300">
                <span className="text-gray-500">Version:</span> <span className="font-semibold">{stats.version}</span>
              </span>
              <span className="text-gray-300">
                <span className="text-gray-500">ID:</span> <span className="font-mono text-xs">{stats.model_id.slice(0, 20)}...</span>
              </span>
            </div>
          </div>
          <div className="text-right">
            <div className="text-sm text-gray-400">Trained on</div>
            <div className="text-white font-semibold">{new Date(stats.training_date).toLocaleDateString()}</div>
          </div>
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Accuracy"
          value={stats.accuracy}
          color="blue"
          icon="🎯"
        />
        <MetricCard
          title="Precision"
          value={stats.precision}
          color="purple"
          icon="🔍"
        />
        <MetricCard
          title="Recall"
          value={stats.recall}
          color="pink"
          icon="📊"
        />
        <MetricCard
          title="F1 Score"
          value={stats.f1_score}
          color="green"
          icon="⭐"
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Bar Chart */}
        <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
          <h4 className="text-lg font-semibold text-white mb-4">Performance Metrics</h4>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={metricsData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="name" stroke="#9ca3af" />
              <YAxis stroke="#9ca3af" domain={[0, 100]} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1f2937',
                  border: '1px solid #374151',
                  borderRadius: '8px',
                  color: '#fff',
                }}
              />
              <Bar dataKey="value" fill="#3b82f6" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Model Performance Summary */}
        <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
          <h4 className="text-lg font-semibold text-white mb-4">Performance Summary</h4>
          <div className="space-y-4">
            {metricsData.map((metric, index) => (
              <div key={metric.name}>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-gray-300 font-medium">{metric.name}</span>
                  <span className="text-white font-bold">{metric.value.toFixed(2)}%</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-3 overflow-hidden">
                  <div
                    className="h-3 rounded-full transition-all duration-500"
                    style={{
                      width: `${metric.value}%`,
                      backgroundColor: COLORS[index],
                    }}
                  ></div>
                </div>
              </div>
            ))}
            <div className="mt-6 pt-4 border-t border-gray-700">
              <div className="text-center">
                <div className="text-sm text-gray-400 mb-1">Overall Performance</div>
                <div className="text-3xl font-bold text-green-400">
                  {((stats.accuracy + stats.precision + stats.recall + stats.f1_score) / 4 * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-gray-500 mt-1">Average of all metrics</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Training Info */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
          <h4 className="text-lg font-semibold text-white mb-4">Training Data</h4>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Training Samples</span>
              <span className="text-white font-semibold">{stats.training_samples.toLocaleString()}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Test Samples</span>
              <span className="text-white font-semibold">{stats.test_samples.toLocaleString()}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Total Samples</span>
              <span className="text-white font-semibold">
                {(stats.training_samples + stats.test_samples).toLocaleString()}
              </span>
            </div>
          </div>
        </div>

        <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
          <h4 className="text-lg font-semibold text-white mb-4">Model Details</h4>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Algorithm</span>
              <span className="text-white font-semibold">{stats.algorithm}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Model Name</span>
              <span className="text-white font-semibold">{stats.model_name}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Version</span>
              <span className="text-white font-semibold">v{stats.version}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function MetricCard({ title, value, color, icon }: { title: string; value: number; color: string; icon: string }) {
  const percentage = (value * 100).toFixed(2);
  
  const colorClasses = {
    blue: 'from-blue-900/50 to-blue-800/50 border-blue-700',
    purple: 'from-purple-900/50 to-purple-800/50 border-purple-700',
    pink: 'from-pink-900/50 to-pink-800/50 border-pink-700',
    green: 'from-green-900/50 to-green-800/50 border-green-700',
  };

  const textColorClasses = {
    blue: 'text-blue-400',
    purple: 'text-purple-400',
    pink: 'text-pink-400',
    green: 'text-green-400',
  };

  return (
    <div className={`bg-gradient-to-br ${colorClasses[color as keyof typeof colorClasses]} backdrop-blur-sm rounded-lg p-6 border card-hover`}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-gray-300 text-sm font-medium">{title}</span>
        <span className="text-2xl">{icon}</span>
      </div>
      <div className={`text-3xl font-bold ${textColorClasses[color as keyof typeof textColorClasses]} mb-1`}>
        {percentage}%
      </div>
      <div className="w-full bg-gray-700 rounded-full h-2 mt-3">
        <div
          className={`h-2 rounded-full ${color === 'blue' ? 'bg-blue-500' : color === 'purple' ? 'bg-purple-500' : color === 'pink' ? 'bg-pink-500' : 'bg-green-500'}`}
          style={{ width: `${percentage}%` }}
        ></div>
      </div>
    </div>
  );
}
