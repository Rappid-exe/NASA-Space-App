'use client';

import type { ClassificationResult } from '@/lib/types';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';

interface ResultsDisplayProps {
  result: ClassificationResult;
}

export default function ResultsDisplay({ result }: ResultsDisplayProps) {
  const isConfirmed = result.prediction === 'CONFIRMED';
  const confidencePercentage = (result.confidence * 100).toFixed(1);

  const chartData = [
    { name: 'Confirmed', value: result.probabilities.CONFIRMED * 100 },
    { name: 'False Positive', value: result.probabilities.FALSE_POSITIVE * 100 },
  ];

  const COLORS = ['#10b981', '#ef4444'];

  return (
    <div className="space-y-8 animate-fadeInScale">
      {/* Main Result Card with refined styling */}
      <div className={`rounded-2xl p-10 border-2 backdrop-blur-md shadow-2xl transition-all duration-slow ${
        isConfirmed 
          ? 'bg-gradient-to-br from-green-900/70 to-emerald-900/70 border-green-400 shadow-glow-green' 
          : 'bg-gradient-to-br from-red-900/70 to-rose-900/70 border-red-400 shadow-glow-red'
      }`}>
        <div className="text-center space-y-4">
          <div className="text-7xl mb-6 animate-fadeInScale">
            {isConfirmed ? '✅' : '❌'}
          </div>
          <h3 className="font-display text-4xl font-bold text-white mb-4 drop-shadow-lg">
            {isConfirmed ? 'Confirmed Exoplanet' : 'False Positive'}
          </h3>
          <div className="text-6xl font-extrabold mb-2">
            <span className={`${isConfirmed ? 'text-green-300' : 'text-red-300'} drop-shadow-lg`}>
              {confidencePercentage}%
            </span>
          </div>
          <p className="text-gray-200 text-xl font-medium">Confidence Score</p>
        </div>
      </div>

      {/* Probability Chart with refined styling */}
      <div className="glass-strong rounded-2xl p-8 border border-white/30 shadow-xl hover:shadow-2xl transition-all duration-normal">
        <h4 className="font-display text-2xl font-semibold text-white mb-6 drop-shadow">Classification Probabilities</h4>
        <ResponsiveContainer width="100%" height={250}>
          <PieChart>
            <Pie
              data={chartData}
              cx="50%"
              cy="50%"
              labelLine={false}
              label={({ name, value }) => `${name}: ${value.toFixed(1)}%`}
              outerRadius={80}
              fill="#8884d8"
              dataKey="value"
            >
              {chartData.map((entry, index) => (
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
              formatter={(value: number) => `${value.toFixed(2)}%`}
            />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </div>

      {/* Detailed Probabilities with refined styling */}
      <div className="glass-strong rounded-2xl p-8 border border-white/30 shadow-xl hover:shadow-2xl transition-all duration-normal">
        <h4 className="font-display text-2xl font-semibold text-white mb-6 drop-shadow">Detailed Probabilities</h4>
        <div className="space-y-6">
          {/* Confirmed Probability with refined styling */}
          <div>
            <div className="flex justify-between items-center mb-3">
              <span className="text-gray-200 text-lg font-medium">Confirmed Exoplanet</span>
              <span className="text-green-300 font-bold text-xl">
                {(result.probabilities.CONFIRMED * 100).toFixed(2)}%
              </span>
            </div>
            <div className="w-full bg-gray-800 rounded-full h-4 overflow-hidden shadow-inner">
              <div
                className="bg-gradient-to-r from-green-500 to-emerald-400 h-4 rounded-full transition-all duration-slower shadow-glow-green"
                style={{ width: `${result.probabilities.CONFIRMED * 100}%` }}
              ></div>
            </div>
          </div>

          {/* False Positive Probability with refined styling */}
          <div>
            <div className="flex justify-between items-center mb-3">
              <span className="text-gray-200 text-lg font-medium">False Positive</span>
              <span className="text-red-300 font-bold text-xl">
                {(result.probabilities.FALSE_POSITIVE * 100).toFixed(2)}%
              </span>
            </div>
            <div className="w-full bg-gray-800 rounded-full h-4 overflow-hidden shadow-inner">
              <div
                className="bg-gradient-to-r from-red-500 to-rose-400 h-4 rounded-full transition-all duration-slower shadow-glow-red"
                style={{ width: `${result.probabilities.FALSE_POSITIVE * 100}%` }}
              ></div>
            </div>
          </div>
        </div>
      </div>

      {/* Explanation with refined styling */}
      <div className="glass-strong border border-blue-500/40 rounded-2xl p-8 shadow-xl hover:shadow-glow-blue transition-all duration-normal">
        <h4 className="font-display text-2xl font-semibold text-blue-300 mb-4 flex items-center gap-3 drop-shadow">
          <span className="text-3xl">💡</span>
          <span>Explanation</span>
        </h4>
        <p className="text-gray-100 text-lg leading-relaxed drop-shadow">
          {result.explanation}
        </p>
      </div>

      {/* Confidence Indicator with refined styling */}
      <div className="glass-strong rounded-2xl p-8 border border-white/30 shadow-xl hover:shadow-2xl transition-all duration-normal">
        <h4 className="font-display text-2xl font-semibold text-white mb-6 drop-shadow">Confidence Level</h4>
        <div className="space-y-2">
          {result.confidence >= 0.9 ? (
            <div className="flex items-center space-x-3">
              <span className="text-3xl">🎯</span>
              <div>
                <div className="text-green-400 font-semibold">Very High Confidence</div>
                <div className="text-sm text-gray-400">The model is very certain about this classification</div>
              </div>
            </div>
          ) : result.confidence >= 0.75 ? (
            <div className="flex items-center space-x-3">
              <span className="text-3xl">✅</span>
              <div>
                <div className="text-blue-400 font-semibold">High Confidence</div>
                <div className="text-sm text-gray-400">The model is confident about this classification</div>
              </div>
            </div>
          ) : result.confidence >= 0.6 ? (
            <div className="flex items-center space-x-3">
              <span className="text-3xl">⚖️</span>
              <div>
                <div className="text-yellow-400 font-semibold">Moderate Confidence</div>
                <div className="text-sm text-gray-400">The model has moderate certainty about this classification</div>
              </div>
            </div>
          ) : (
            <div className="flex items-center space-x-3">
              <span className="text-3xl">⚠️</span>
              <div>
                <div className="text-orange-400 font-semibold">Low Confidence</div>
                <div className="text-sm text-gray-400">The model is uncertain - consider additional analysis</div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
