'use client';

import { useEffect, useState } from 'react';
import { getDatasetComparison } from '@/lib/api';
import type { DatasetComparison } from '@/lib/types';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

export default function DatasetComparisonView() {
  const [comparison, setComparison] = useState<DatasetComparison | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchComparison() {
      try {
        const data = await getDatasetComparison();
        setComparison(data);
      } catch (err) {
        setError('Failed to load dataset comparison.');
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    fetchComparison();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-400">Loading dataset comparison...</p>
        </div>
      </div>
    );
  }

  if (error || !comparison || comparison.missions.length === 0) {
    return (
      <div className="bg-yellow-900/20 border border-yellow-700 rounded-lg p-6 text-center">
        <p className="text-yellow-400">⚠️ {error || 'No dataset comparison data available'}</p>
        <p className="text-sm text-gray-400 mt-2">Ensure datasets are downloaded in the data/raw directory</p>
      </div>
    );
  }

  const chartData = comparison.missions.map((m) => ({
    name: m.name,
    confirmed: m.confirmed_exoplanets,
    false_positives: m.false_positives,
    candidates: m.candidates,
    total: m.total_observations,
  }));

  const pieData = comparison.missions.map((m) => ({
    name: m.name,
    value: m.confirmed_exoplanets,
  }));

  const COLORS = ['#3b82f6', '#8b5cf6', '#ec4899'];

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-900/50 to-purple-900/50 backdrop-blur-sm rounded-lg p-6 border border-blue-800">
        <h2 className="text-2xl font-bold text-white mb-2">NASA Mission Dataset Comparison</h2>
        <p className="text-gray-300">
          Comparing exoplanet discoveries across Kepler, TESS, and K2 missions
        </p>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-gradient-to-br from-blue-900/50 to-blue-800/50 backdrop-blur-sm rounded-lg p-6 border border-blue-700">
          <div className="text-sm text-gray-300 mb-2">Total Observations</div>
          <div className="text-3xl font-bold text-blue-400">
            {comparison.summary.total_observations.toLocaleString()}
          </div>
        </div>
        <div className="bg-gradient-to-br from-green-900/50 to-green-800/50 backdrop-blur-sm rounded-lg p-6 border border-green-700">
          <div className="text-sm text-gray-300 mb-2">Confirmed Exoplanets</div>
          <div className="text-3xl font-bold text-green-400">
            {comparison.summary.total_confirmed.toLocaleString()}
          </div>
        </div>
        <div className="bg-gradient-to-br from-red-900/50 to-red-800/50 backdrop-blur-sm rounded-lg p-6 border border-red-700">
          <div className="text-sm text-gray-300 mb-2">False Positives</div>
          <div className="text-3xl font-bold text-red-400">
            {comparison.summary.total_false_positives.toLocaleString()}
          </div>
        </div>
      </div>

      {/* Mission Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {comparison.missions.map((mission, index) => (
          <MissionCard key={mission.name} mission={mission} color={COLORS[index]} />
        ))}
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Bar Chart */}
        <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
          <h3 className="text-xl font-semibold text-white mb-4">Observations by Mission</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="name" stroke="#9ca3af" />
              <YAxis stroke="#9ca3af" />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1f2937',
                  border: '1px solid #374151',
                  borderRadius: '8px',
                  color: '#fff',
                }}
              />
              <Legend />
              <Bar dataKey="confirmed" fill="#10b981" name="Confirmed" />
              <Bar dataKey="false_positives" fill="#ef4444" name="False Positives" />
              <Bar dataKey="candidates" fill="#f59e0b" name="Candidates" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Pie Chart */}
        <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
          <h3 className="text-xl font-semibold text-white mb-4">Confirmed Exoplanets Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, value }) => `${name}: ${value}`}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
              >
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1f2937',
                  border: '1px solid #374151',
                  borderRadius: '8px',
                  color: '#fff',
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Mission Information */}
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <h3 className="text-xl font-semibold text-white mb-4">About the Missions</h3>
        <div className="space-y-4">
          <MissionInfo
            name="Kepler Space Telescope"
            years="2009-2018"
            description="Continuously monitored 150,000 stars in a single field of view to detect transiting exoplanets. Discovered over 2,600 confirmed exoplanets."
            icon="🔭"
          />
          <MissionInfo
            name="TESS (Transiting Exoplanet Survey Satellite)"
            years="2018-present"
            description="All-sky survey focusing on nearby bright stars. Designed to find exoplanets around stars close enough for follow-up observations."
            icon="🛰️"
          />
          <MissionInfo
            name="K2 Mission"
            years="2014-2018"
            description="Extended Kepler mission observing different fields along the ecliptic plane. Discovered over 500 confirmed exoplanets."
            icon="🌌"
          />
        </div>
      </div>
    </div>
  );
}

function MissionCard({ mission, color }: { mission: any; color: string }) {
  return (
    <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700 card-hover">
      <h3 className="text-xl font-bold text-white mb-4">{mission.name}</h3>
      <div className="space-y-3">
        <div className="flex justify-between items-center">
          <span className="text-gray-400">Total Observations</span>
          <span className="text-white font-semibold">{mission.total_observations.toLocaleString()}</span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-gray-400">Confirmed</span>
          <span className="text-green-400 font-semibold">{mission.confirmed_exoplanets.toLocaleString()}</span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-gray-400">False Positives</span>
          <span className="text-red-400 font-semibold">{mission.false_positives.toLocaleString()}</span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-gray-400">Candidates</span>
          <span className="text-yellow-400 font-semibold">{mission.candidates.toLocaleString()}</span>
        </div>
        <div className="pt-3 border-t border-gray-700">
          <div className="flex justify-between items-center">
            <span className="text-gray-400">Confirmation Rate</span>
            <span className="text-blue-400 font-semibold">{mission.confirmation_rate.toFixed(2)}%</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function MissionInfo({ name, years, description, icon }: { name: string; years: string; description: string; icon: string }) {
  return (
    <div className="bg-gray-700/50 rounded-lg p-4 border border-gray-600">
      <div className="flex items-start space-x-3">
        <span className="text-3xl">{icon}</span>
        <div className="flex-1">
          <div className="flex items-center justify-between mb-2">
            <h4 className="font-semibold text-white">{name}</h4>
            <span className="text-sm text-gray-400">{years}</span>
          </div>
          <p className="text-sm text-gray-300">{description}</p>
        </div>
      </div>
    </div>
  );
}
