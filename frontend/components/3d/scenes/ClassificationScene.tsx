'use client'

import { useMemo } from 'react'
import { Canvas } from '@react-three/fiber'
import ClassificationPlanet from '../objects/ClassificationPlanet'
import Starfield from '../objects/Starfield'
import ZoomAnimation from '../controls/ZoomAnimation'
import ParticleSystem from '../effects/ParticleSystem'
import PostProcessing from '../effects/PostProcessing'

interface ClassificationSceneProps {
  result?: 'CONFIRMED' | 'FALSE_POSITIVE' | null
  isClassifying?: boolean
  className?: string
  showParticles?: boolean
}

// Scene content component
function SceneContent({
  result,
  isClassifying,
  showParticles = true,
}: Omit<ClassificationSceneProps, 'className'>) {
  // Detect if mobile for performance optimization
  const isMobile = useMemo(() => {
    if (typeof window === 'undefined') return false
    return window.innerWidth < 768
  }, [])

  // Adjust quality based on device
  const sceneConfig = useMemo(
    () => ({
      starCount: isMobile ? 1000 : 3000,
      particleCount: isMobile ? 50 : 100,
      enableShadows: !isMobile,
      enablePostProcessing: !isMobile,
    }),
    [isMobile]
  )

  // Determine particle color based on result
  const particleColor = result === 'CONFIRMED' ? '#4A90E2' : '#E27B58'

  return (
    <>
      {/* Enhanced lighting setup for dramatic effect */}
      <ambientLight intensity={0.25} color="#1a1f3a" />
      <hemisphereLight intensity={0.3} color="#ffffff" groundColor="#0a0e27" />
      
      {/* Key light for dramatic shadows */}
      <directionalLight 
        position={[5, 5, 5]} 
        intensity={0.8}
        color="#ffffff"
        castShadow={sceneConfig.enableShadows}
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
        shadow-camera-far={50}
        shadow-camera-left={-10}
        shadow-camera-right={10}
        shadow-camera-top={10}
        shadow-camera-bottom={-10}
      />
      
      {/* Fill light for softer shadows */}
      <directionalLight 
        position={[-3, 2, -3]} 
        intensity={0.3}
        color="#4A90E2"
      />

      {/* Classification Planet */}
      <ClassificationPlanet result={result} isClassifying={isClassifying} />

      {/* Particle effects for successful classification */}
      {showParticles && result && (
        <ParticleSystem
          active={!!result}
          count={sceneConfig.particleCount}
          color={particleColor}
          size={0.08}
          spread={4}
        />
      )}

      {/* Starfield background */}
      <Starfield count={sceneConfig.starCount} radius={50} depth={30} />

      {/* Camera with zoom animation */}
      <ZoomAnimation
        startPosition={[0, 0, 20]}
        endPosition={[0, 0, 8]}
        duration={1500}
        autoStart={true}
      />

      {/* Post-processing effects - Temporarily disabled due to initialization issue */}
      {/* {sceneConfig.enablePostProcessing && (
        <PostProcessing
          enableBloom={true}
          enableVignette={true}
          enableDepthOfField={true}
          bloomIntensity={2.0}
          bloomLuminanceThreshold={0.8}
          dofFocusDistance={0.02}
          dofFocalLength={0.05}
          dofBokehScale={2}
          vignetteOffset={0.5}
          vignetteDarkness={0.6}
        />
      )} */}
    </>
  )
}

// Main Classification Scene wrapper component
export default function ClassificationScene({
  result = null,
  isClassifying = false,
  className = '',
  showParticles = true,
}: ClassificationSceneProps) {
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
        <SceneContent
          result={result}
          isClassifying={isClassifying}
          showParticles={showParticles}
        />
      </Canvas>
    </div>
  )
}
