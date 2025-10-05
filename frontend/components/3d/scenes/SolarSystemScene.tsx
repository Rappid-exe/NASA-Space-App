'use client'

import { useRef, useMemo } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { PerspectiveCamera } from '@react-three/drei'
import { Group } from 'three'
import Sun from '../objects/Sun'
import Planet from '../objects/Planet'
import Starfield from '../objects/Starfield'
import OrbitPath from '../objects/OrbitPath'
import PostProcessing from '../effects/PostProcessing'

// Planet configurations with realistic orbital distances, speeds, and starting positions
const PLANET_CONFIGS = [
  {
    name: 'mercury',
    distance: 8,
    size: 0.4,
    speed: 0.04,
    color: '#8C7853',
    startAngle: 0,
  },
  {
    name: 'venus',
    distance: 12,
    size: 0.9,
    speed: 0.03,
    color: '#FFC649',
    startAngle: Math.PI * 0.3,
  },
  {
    name: 'earth',
    distance: 16,
    size: 1,
    speed: 0.02,
    color: '#4A90E2',
    startAngle: Math.PI * 0.6,
  },
  {
    name: 'mars',
    distance: 20,
    size: 0.5,
    speed: 0.015,
    color: '#E27B58',
    startAngle: Math.PI * 0.9,
  },
  {
    name: 'jupiter',
    distance: 28,
    size: 2,
    speed: 0.008,
    color: '#C88B3A',
    startAngle: Math.PI * 1.2,
  },
  {
    name: 'saturn',
    distance: 36,
    size: 1.8,
    speed: 0.006,
    color: '#FAD5A5',
    startAngle: Math.PI * 1.5,
  },
  {
    name: 'uranus',
    distance: 44,
    size: 1.2,
    speed: 0.004,
    color: '#4FD0E7',
    startAngle: Math.PI * 1.7,
  },
  {
    name: 'neptune',
    distance: 52,
    size: 1.2,
    speed: 0.003,
    color: '#4166F5',
    startAngle: Math.PI * 1.9,
  },
]

// Camera drift controller component
function CameraController() {
  const cameraRef = useRef<any>(null)

  useFrame((state) => {
    if (cameraRef.current) {
      // Refined camera drift with elevated angle to view orbits
      const time = state.clock.elapsedTime * 0.08
      
      // Elevated position to see orbital paths from above at an angle
      cameraRef.current.position.x = Math.sin(time) * 4 + Math.sin(time * 1.3) * 2
      cameraRef.current.position.y = 25 + Math.sin(time * 0.6) * 3 // Elevated to see orbits
      cameraRef.current.position.z = Math.cos(time) * 4 + Math.cos(time * 1.1) * 2 + 50
      
      // Look at center (sun) with slight offset
      const lookAtX = Math.sin(time * 0.5) * 1
      const lookAtY = Math.cos(time * 0.3) * 0.5
      cameraRef.current.lookAt(lookAtX, lookAtY, 0)
    }
  })

  return (
    <PerspectiveCamera
      ref={cameraRef}
      makeDefault
      position={[0, 25, 50]}
      fov={60}
      near={0.1}
      far={1000}
    />
  )
}

// Scene content component (separated for performance)
function SceneContent() {
  const planetsRef = useRef<Group>(null)

  // Detect if mobile for performance optimization
  const isMobile = useMemo(() => {
    if (typeof window === 'undefined') return false
    return window.innerWidth < 768
  }, [])

  // Adjust quality based on device
  const sceneConfig = useMemo(() => ({
    starCount: isMobile ? 2000 : 5000,
    planetSegments: isMobile ? 32 : 64,
    sunSize: isMobile ? 1.5 : 2,
    enableShadows: !isMobile,
    enablePostProcessing: !isMobile,
  }), [isMobile])

  return (
    <>
      {/* Enhanced lighting setup for realistic atmosphere */}
      <ambientLight intensity={0.15} color="#1a1f3a" />
      <hemisphereLight intensity={0.2} color="#ffffff" groundColor="#0a0e27" />
      
      {/* Sun with enhanced point light */}
      <Sun size={sceneConfig.sunSize} intensity={2.5} />

      {/* Orbital path rings */}
      <group>
        {PLANET_CONFIGS.map((config) => (
          <OrbitPath
            key={`orbit-${config.name}`}
            radius={config.distance}
            color={config.color}
            opacity={0.2}
          />
        ))}
      </group>

      {/* Planets group */}
      <group ref={planetsRef}>
        {PLANET_CONFIGS.map((config) => (
          <group key={config.name} rotation={[0, config.startAngle, 0]}>
            <Planet
              name={config.name}
              distance={config.distance}
              size={config.size}
              speed={config.speed}
              color={config.color}
              segments={sceneConfig.planetSegments}
            />
          </group>
        ))}
      </group>

      {/* Starfield background */}
      <Starfield count={sceneConfig.starCount} radius={100} depth={50} />

      {/* Camera with drift animation */}
      <CameraController />

      {/* Post-processing effects - Temporarily disabled due to initialization issue */}
      {/* {sceneConfig.enablePostProcessing && (
        <PostProcessing
          enableBloom={true}
          enableVignette={true}
          enableDepthOfField={false}
          bloomIntensity={1.5}
          bloomLuminanceThreshold={0.9}
          vignetteOffset={0.5}
          vignetteDarkness={0.5}
        />
      )} */}
    </>
  )
}

// Main Solar System Scene wrapper component
interface SolarSystemSceneProps {
  className?: string
}

export default function SolarSystemScene({ className = '' }: SolarSystemSceneProps) {
  // Detect if mobile for shadow configuration
  const isMobile = useMemo(() => {
    if (typeof window === 'undefined') return false
    return window.innerWidth < 768
  }, [])

  return (
    <div className={`w-full h-full ${className}`}>
      <Canvas
        shadows={!isMobile} // Enable shadows on desktop
        gl={{
          antialias: true,
          alpha: true,
          powerPreference: 'high-performance',
        }}
        dpr={[1, 2]} // Limit pixel ratio for performance
        performance={{ min: 0.5 }} // Adaptive performance
      >
        <SceneContent />
      </Canvas>
    </div>
  )
}
