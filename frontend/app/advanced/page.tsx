'use client';

import { useState } from 'react';
import Link from 'next/link';
import FeatureImportanceView from '@/components/FeatureImportanceView';
import DatasetComparisonView from '@/components/DatasetComparisonView';
import HyperparameterTuning from '@/components/HyperparameterTuning';
import ModelRetraining from '@/components/ModelRetraining';
import ExoplanetEducation from '@/components/ExoplanetEducation';

type TabType = 'feature-importance' | 'dataset-comparison' | 'hyperparameter-tuning' | 'retraining' | 'education';

export default function AdvancedPage() {
  const [activeTab, setActiveTab] = useState<TabType>('feature-importance');

  const tabs = [
    { id: 'feature-importance' as TabType, label: 'Feature Importance', icon: '📊' },
    { id: 'dataset-comparison' as TabType, label: 'Dataset Comparison', icon: '🔬' },
    { id: 'hyperparameter-tuning' as TabType, label: 'Hyperparameter Tuning', icon: '⚙️' },
    { id: 'retraining' as TabType, label: 'Model Retraining', icon: '🔄' },
    { id: 'education' as TabType, label: 'Learn About Exoplanets', icon: '📚' },
  ];

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
                  <h1 className="text-2xl font-bold text-white">Advanced Features</h1>
                  <p className="text-sm text-gray-400">Model Optimization & Analysis</p>
                </div>
              </Link>
            </div>
            
            <Link
              href="/"
              className="px-4 py-2 bg-gray-800 text-white rounded-lg hover:bg-gray-700 transition-colors border border-gray-700"
            >
              ← Back to Home
            </Link>
          </div>
        </div>
      </header>

      {/* Tab Navigation */}
      <div className="container mx-auto px-4 py-6">
        <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-2 border border-gray-700 flex flex-wrap gap-2">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center space-x-2 px-4 py-3 rounded-lg font-medium transition-all ${
                activeTab === tab.id
                  ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg'
                  : 'bg-gray-700/50 text-gray-300 hover:bg-gray-700'
              }`}
            >
              <span className="text-xl">{tab.icon}</span>
              <span className="hidden sm:inline">{tab.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Tab Content */}
      <div className="container mx-auto px-4 py-6">
        {activeTab === 'feature-importance' && <FeatureImportanceView />}
        {activeTab === 'dataset-comparison' && <DatasetComparisonView />}
        {activeTab === 'hyperparameter-tuning' && <HyperparameterTuning />}
        {activeTab === 'retraining' && <ModelRetraining />}
        {activeTab === 'education' && <ExoplanetEducation />}
      </div>
    </div>
  );
}
