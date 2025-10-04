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
    <div className="space-y-6 animate-fadeIn">
      {/* Main Result Card */}
      <div className={`rounded-lg p-8 border-2 ${
        isConfirmed 
          ? 'bg-gradient-to-br from-green-900/50 to-emerald-900/50 border-green-500' 
          : 'bg-gradient-to-br from-red-900/50 to-rose-900/50 border-red-500'
      }`}>
        <div className="text-center">
          <div className="text-6xl mb-4">
            {isConfirmed ? '✅' : '❌'}
          </div>
          <h3 className="text-3xl font-bold text-white mb-2">
            {isConfirmed ? 'Confirmed Exoplanet' : 'False Positive'}
          </h3>
          <div className="text-5xl font-bold mb-4">
            <span className={isConfirmed ? 'text-green-400' : 'text-red-400'}>
              {confidencePercentage}%
            </span>
          </div>
          <p className="text-gray-300 text-lg">Confidence Score</p>
        </div>
      </div>

      {/* Probability Chart */}
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <h4 className="text-lg font-semibold text-white mb-4">Classification Probabilities</h4>
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

      {/* Detailed Probabilities */}
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <h4 className="text-lg font-semibold text-white mb-4">Detailed Probabilities</h4>
        <div className="space-y-4">
          {/* Confirmed Probability */}
          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-gray-300">Confirmed Exoplanet</span>
              <span className="text-green-400 font-semibold">
                {(result.probabilities.CONFIRMED * 100).toFixed(2)}%
              </span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-3">
              <div
                className="bg-green-500 h-3 rounded-full transition-all duration-500"
                style={{ width: `${result.probabilities.CONFIRMED * 100}%` }}
              ></div>
            </div>
          </div>

          {/* False Positive Probability */}
          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-gray-300">False Positive</span>
              <span className="text-red-400 font-semibold">
                {(result.probabilities.FALSE_POSITIVE * 100).toFixed(2)}%
              </span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-3">
              <div
                className="bg-red-500 h-3 rounded-full transition-all duration-500"
                style={{ width: `${result.probabilities.FALSE_POSITIVE * 100}%` }}
              ></div>
            </div>
          </div>
        </div>
      </div>

      {/* Explanation */}
      <div className="bg-blue-900/20 border border-blue-800 rounded-lg p-6">
        <h4 className="text-lg font-semibold text-blue-400 mb-3 flex items-center">
          <span className="mr-2">💡</span>
          Explanation
        </h4>
        <p className="text-gray-300 leading-relaxed">
          {result.explanation}
        </p>
      </div>

      {/* Confidence Indicator */}
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <h4 className="text-lg font-semibold text-white mb-4">Confidence Level</h4>
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
