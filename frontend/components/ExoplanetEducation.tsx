'use client';

import { useEffect, useState } from 'react';
import { getExoplanetEducation } from '@/lib/api';
import type { ExoplanetEducation as ExoplanetEducationType } from '@/lib/types';

export default function ExoplanetEducation() {
  const [education, setEducation] = useState<ExoplanetEducationType | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeSection, setActiveSection] = useState<string>('overview');

  useEffect(() => {
    async function fetchEducation() {
      try {
        const data = await getExoplanetEducation();
        setEducation(data);
      } catch (err) {
        setError('Failed to load educational content.');
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    fetchEducation();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-400">Loading educational content...</p>
        </div>
      </div>
    );
  }

  if (error || !education) {
    return (
      <div className="bg-yellow-900/20 border border-yellow-700 rounded-lg p-6 text-center">
        <p className="text-yellow-400">⚠️ {error || 'No educational content available'}</p>
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-900/50 to-purple-900/50 backdrop-blur-sm rounded-lg p-6 border border-blue-800">
        <h2 className="text-2xl font-bold text-white mb-2">Learn About Exoplanets</h2>
        <p className="text-gray-300">
          Understanding the science behind exoplanet discovery and classification
        </p>
      </div>

      {/* Navigation */}
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-2 border border-gray-700 flex flex-wrap gap-2">
        <NavButton
          active={activeSection === 'overview'}
          onClick={() => setActiveSection('overview')}
          icon="🌍"
          label="Overview"
        />
        <NavButton
          active={activeSection === 'detection'}
          onClick={() => setActiveSection('detection')}
          icon="🔭"
          label="Detection Methods"
        />
        <NavButton
          active={activeSection === 'types'}
          onClick={() => setActiveSection('types')}
          icon="🪐"
          label="Planet Types"
        />
        <NavButton
          active={activeSection === 'features'}
          onClick={() => setActiveSection('features')}
          icon="📊"
          label="Features"
        />
        <NavButton
          active={activeSection === 'missions'}
          onClick={() => setActiveSection('missions')}
          icon="🛰️"
          label="Missions"
        />
      </div>

      {/* Content Sections */}
      {activeSection === 'overview' && (
        <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
          <h3 className="text-2xl font-semibold text-white mb-4">What Are Exoplanets?</h3>
          <p className="text-gray-300 text-lg leading-relaxed">{education.overview}</p>
          
          <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
            <StatCard icon="🌟" value="5,000+" label="Confirmed Exoplanets" />
            <StatCard icon="🔬" value="9,000+" label="Candidate Exoplanets" />
            <StatCard icon="🌌" value="3,800+" label="Planetary Systems" />
          </div>
        </div>
      )}

      {activeSection === 'detection' && (
        <div className="space-y-4">
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
            <h3 className="text-2xl font-semibold text-white mb-4">Detection Methods</h3>
            {Object.entries(education.detection_methods).map(([key, method]: [string, any]) => (
              <div key={key} className="bg-gray-700/50 rounded-lg p-4 border border-gray-600">
                <h4 className="text-xl font-semibold text-blue-400 mb-2">{method.name}</h4>
                <p className="text-gray-300 mb-3">{method.description}</p>
                <div className="flex flex-wrap gap-2">
                  {method.key_features.map((feature: string) => (
                    <span
                      key={feature}
                      className="px-3 py-1 bg-blue-900/30 border border-blue-700 rounded-full text-sm text-blue-300"
                    >
                      {feature.replace(/_/g, ' ')}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>

          <div className="bg-blue-900/20 border border-blue-700 rounded-lg p-6">
            <h4 className="text-lg font-semibold text-blue-400 mb-3">💡 How Transit Detection Works</h4>
            <ol className="space-y-2 text-gray-300 list-decimal list-inside">
              <li>A telescope continuously monitors the brightness of a star</li>
              <li>When a planet passes in front of the star, it blocks some light</li>
              <li>This causes a small, periodic dip in the star's brightness</li>
              <li>By analyzing these dips, we can determine the planet's size, orbit, and more</li>
            </ol>
          </div>
        </div>
      )}

      {activeSection === 'types' && (
        <div className="space-y-4">
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
            <h3 className="text-2xl font-semibold text-white mb-4">Types of Exoplanets</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {Object.entries(education.planet_types).map(([key, type]: [string, any]) => (
                <PlanetTypeCard key={key} type={type} />
              ))}
            </div>
          </div>

          <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
            <h4 className="text-lg font-semibold text-white mb-3">Size Comparison</h4>
            <div className="space-y-3">
              <SizeBar label="Earth-like" percentage={15} color="green" />
              <SizeBar label="Super-Earth" percentage={35} color="blue" />
              <SizeBar label="Neptune-like" percentage={60} color="purple" />
              <SizeBar label="Jupiter-like" percentage={100} color="orange" />
            </div>
          </div>
        </div>
      )}

      {activeSection === 'features' && (
        <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
          <h3 className="text-2xl font-semibold text-white mb-4">Understanding the Features</h3>
          <div className="space-y-4">
            {Object.entries(education.features_explained).map(([key, description]) => (
              <FeatureExplanation
                key={key}
                name={key.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())}
                description={description}
              />
            ))}
          </div>
        </div>
      )}

      {activeSection === 'missions' && (
        <div className="space-y-4">
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
            <h3 className="text-2xl font-semibold text-white mb-4">NASA Exoplanet Missions</h3>
            <div className="space-y-4">
              {Object.entries(education.missions).map(([key, mission]: [string, any]) => (
                <MissionCard key={key} mission={mission} />
              ))}
            </div>
          </div>

          <div className="bg-purple-900/20 border border-purple-700 rounded-lg p-6">
            <h4 className="text-lg font-semibold text-purple-400 mb-3">🚀 The Future of Exoplanet Discovery</h4>
            <p className="text-gray-300 mb-3">
              Future missions like the James Webb Space Telescope (JWST) and the Nancy Grace Roman Space
              Telescope will revolutionize our understanding of exoplanets by:
            </p>
            <ul className="space-y-2 text-gray-300 list-disc list-inside">
              <li>Analyzing exoplanet atmospheres for signs of life</li>
              <li>Discovering smaller, Earth-like planets in habitable zones</li>
              <li>Characterizing the composition and weather of distant worlds</li>
              <li>Expanding our search to more distant and diverse star systems</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

function NavButton({ active, onClick, icon, label }: { active: boolean; onClick: () => void; icon: string; label: string }) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-all ${
        active
          ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white'
          : 'bg-gray-700/50 text-gray-300 hover:bg-gray-700'
      }`}
    >
      <span>{icon}</span>
      <span className="hidden sm:inline">{label}</span>
    </button>
  );
}

function StatCard({ icon, value, label }: { icon: string; value: string; label: string }) {
  return (
    <div className="bg-gray-700/50 rounded-lg p-4 border border-gray-600 text-center">
      <div className="text-3xl mb-2">{icon}</div>
      <div className="text-2xl font-bold text-blue-400 mb-1">{value}</div>
      <div className="text-sm text-gray-400">{label}</div>
    </div>
  );
}

function PlanetTypeCard({ type }: { type: any }) {
  return (
    <div className="bg-gray-700/50 rounded-lg p-4 border border-gray-600">
      <h4 className="text-lg font-semibold text-white mb-2">{type.name}</h4>
      <div className="space-y-2 text-sm">
        <p className="text-gray-400">
          <span className="font-semibold text-blue-400">Size:</span> {type.radius_range}
        </p>
        <p className="text-gray-300">{type.description}</p>
        <p className="text-gray-400">
          <span className="font-semibold text-green-400">Habitability:</span> {type.habitability}
        </p>
      </div>
    </div>
  );
}

function FeatureExplanation({ name, description }: { name: string; description: string }) {
  return (
    <div className="bg-gray-700/50 rounded-lg p-4 border border-gray-600">
      <h4 className="font-semibold text-blue-400 mb-2">{name}</h4>
      <p className="text-gray-300 text-sm">{description}</p>
    </div>
  );
}

function MissionCard({ mission }: { mission: any }) {
  return (
    <div className="bg-gray-700/50 rounded-lg p-5 border border-gray-600">
      <div className="flex items-start justify-between mb-3">
        <h4 className="text-xl font-semibold text-white">{mission.name}</h4>
        <span className="text-sm text-gray-400 bg-gray-800 px-3 py-1 rounded-full">{mission.years}</span>
      </div>
      <p className="text-gray-300 mb-3">{mission.focus}</p>
      <div className="flex items-center space-x-2">
        <span className="text-green-400 font-semibold">{mission.discoveries}</span>
        <span className="text-gray-500">•</span>
        <span className="text-gray-400 text-sm">discovered</span>
      </div>
    </div>
  );
}

function SizeBar({ label, percentage, color }: { label: string; percentage: number; color: string }) {
  const colorClasses = {
    green: 'bg-green-500',
    blue: 'bg-blue-500',
    purple: 'bg-purple-500',
    orange: 'bg-orange-500',
  };

  return (
    <div>
      <div className="flex justify-between text-sm text-gray-400 mb-1">
        <span>{label}</span>
        <span>Relative Size</span>
      </div>
      <div className="w-full bg-gray-700 rounded-full h-6">
        <div
          className={`h-6 rounded-full ${colorClasses[color as keyof typeof colorClasses]} flex items-center justify-end pr-2`}
          style={{ width: `${percentage}%` }}
        >
          <span className="text-xs text-white font-semibold">{percentage}%</span>
        </div>
      </div>
    </div>
  );
}
