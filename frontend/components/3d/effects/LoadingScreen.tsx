'use client'

import { useProgress } from '@react-three/drei'
import { useEffect, useState } from 'react'

interface LoadingScreenProps {
  onLoadComplete?: () => void
}

export default function LoadingScreen({ onLoadComplete }: LoadingScreenProps) {
  const { active, progress, errors, item, loaded, total } = useProgress()
  const [isVisible, setIsVisible] = useState(true)

  useEffect(() => {
    if (!active && progress === 100) {
      // Delay hiding to allow smooth transition
      const timer = setTimeout(() => {
        setIsVisible(false)
        onLoadComplete?.()
      }, 500)
      return () => clearTimeout(timer)
    }
    
    // Fallback: hide after 3 seconds regardless of progress
    const fallbackTimer = setTimeout(() => {
      setIsVisible(false)
      onLoadComplete?.()
    }, 3000)
    
    return () => clearTimeout(fallbackTimer)
  }, [active, progress, onLoadComplete])

  if (!isVisible) return null

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black transition-opacity duration-500 pointer-events-none"
      style={{ opacity: active ? 1 : 0 }}
    >
      <div className="text-center">
        {/* Loading spinner */}
        <div className="relative w-24 h-24 mx-auto mb-6">
          <div className="absolute inset-0 border-4 border-blue-500/20 rounded-full"></div>
          <div
            className="absolute inset-0 border-4 border-transparent border-t-blue-500 rounded-full animate-spin"
            style={{
              animationDuration: '1s',
            }}
          ></div>
          {/* Inner glow effect */}
          <div className="absolute inset-2 bg-blue-500/10 rounded-full blur-xl"></div>
        </div>

        {/* Loading text */}
        <h2 className="text-2xl font-bold text-white mb-2">
          Loading 3D Scene
        </h2>

        {/* Progress bar */}
        <div className="w-64 h-2 bg-gray-800 rounded-full overflow-hidden mx-auto mb-2">
          <div
            className="h-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-300 ease-out"
            style={{ width: `${progress}%` }}
          ></div>
        </div>

        {/* Progress percentage */}
        <p className="text-gray-400 text-sm">
          {Math.round(progress)}% ({loaded} / {total})
        </p>

        {/* Current item being loaded */}
        {item && (
          <p className="text-gray-500 text-xs mt-2 max-w-xs truncate">
            Loading: {item}
          </p>
        )}

        {/* Error display */}
        {errors.length > 0 && (
          <div className="mt-4 text-red-400 text-sm">
            <p>Error loading assets:</p>
            {errors.map((error, i) => (
              <p key={i} className="text-xs">
                {error}
              </p>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
