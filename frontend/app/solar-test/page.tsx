'use client'

import SolarSystemScene from '@/components/3d/scenes/SolarSystemScene'

export default function SolarTestPage() {
  return (
    <div className="relative w-full h-screen bg-black">
      {/* 3D Solar System Scene */}
      <SolarSystemScene className="absolute inset-0" />
      
      {/* Overlay UI for testing */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="container mx-auto px-4 h-full flex flex-col justify-center items-center">
          <div className="text-center text-white pointer-events-auto">
            <h1 className="text-5xl font-bold mb-4">Solar System Scene Test</h1>
            <p className="text-xl mb-8 text-gray-300">
              Watch the planets orbit and the camera drift
            </p>
            <div className="space-x-4">
              <button className="px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg transition">
                Test Button 1
              </button>
              <button className="px-6 py-3 bg-purple-600 hover:bg-purple-700 rounded-lg transition">
                Test Button 2
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
