'use client'

import { useRef } from 'react'
import { Mesh } from 'three'
import { useFrame } from '@react-three/fiber'
import { useSpring, animated } from '@react-spring/three'

interface ClassificationPlanetProps {
  result?: 'CONFIRMED' | 'FALSE_POSITIVE' | null
  isClassifying?: boolean
}

// Enhanced color configurations for different states
const COLORS = {
  idle: '#9CA3AF', // Refined grey
  classifying: '#9CA3AF', // Grey with pulsing
  CONFIRMED: '#10B981', // Vibrant green
  FALSE_POSITIVE: '#EF4444', // Vibrant red
}

const EMISSIVE_COLORS = {
  idle: '#4B5563',
  classifying: '#6B7280',
  CONFIRMED: '#059669',
  FALSE_POSITIVE: '#DC2626',
}

export default function ClassificationPlanet({
  result = null,
  isClassifying = false,
}: ClassificationPlanetProps) {
  const planetRef = useRef<Mesh>(null)
  const glowRef = useRef<Mesh>(null)

  // Determine current state
  const currentState = isClassifying ? 'classifying' : result || 'idle'

  // Animated color transition with refined easing
  const { color, emissiveColor, emissiveIntensity, scale } = useSpring({
    color: COLORS[currentState],
    emissiveColor: EMISSIVE_COLORS[currentState],
    emissiveIntensity: isClassifying ? 0.9 : result ? 1.5 : 0.4,
    scale: result ? 1.05 : 1,
    config: { 
      tension: 180, 
      friction: 20,
      duration: 600,
    },
  })

  // Rotation speed based on state with smoother transitions
  const rotationSpeed = isClassifying ? 1.2 : result ? 0.6 : 0.4

  // Refined animation loop with smoother timing
  useFrame((state, delta) => {
    const time = state.clock.elapsedTime

    if (planetRef.current) {
      // Self-rotation with variable speed and smooth easing
      planetRef.current.rotation.y += delta * rotationSpeed
      
      // Subtle wobble for organic feel
      planetRef.current.rotation.x = Math.sin(time * 0.3) * 0.03
      planetRef.current.rotation.z = Math.cos(time * 0.4) * 0.02
    }

    if (glowRef.current) {
      // Enhanced pulsing effect during classification
      if (isClassifying) {
        const pulse = Math.sin(time * 2.5) * 0.15 + 1
        glowRef.current.scale.setScalar(pulse * 1.35)
        // Rotate glow for dynamic effect
        glowRef.current.rotation.z = time * 0.5
      } else if (result) {
        // Gentle breathing effect after classification
        const breathe = Math.sin(time * 0.8) * 0.05 + 1
        glowRef.current.scale.setScalar(breathe * 1.35)
      } else {
        glowRef.current.scale.setScalar(1.3)
      }
    }
  })

  return (
    <group>
      {/* Main planet sphere with enhanced materials and shadows */}
      <animated.mesh ref={planetRef} castShadow receiveShadow scale={scale}>
        <sphereGeometry args={[2, 128, 128]} />
        <animated.meshStandardMaterial
          color={color}
          emissive={emissiveColor}
          emissiveIntensity={emissiveIntensity}
          roughness={0.7}
          metalness={0.25}
          toneMapped={false}
          envMapIntensity={0.4}
        />
      </animated.mesh>

      {/* Inner glow layer with refined opacity */}
      <animated.mesh ref={glowRef}>
        <sphereGeometry args={[2.15, 64, 64]} />
        <animated.meshBasicMaterial
          color={emissiveColor}
          transparent
          opacity={isClassifying ? 0.35 : result ? 0.3 : 0.15}
          toneMapped={false}
          depthWrite={false}
        />
      </animated.mesh>

      {/* Middle glow layer */}
      <animated.mesh>
        <sphereGeometry args={[2.35, 48, 48]} />
        <animated.meshBasicMaterial
          color={emissiveColor}
          transparent
          opacity={isClassifying ? 0.2 : result ? 0.18 : 0.08}
          toneMapped={false}
          depthWrite={false}
        />
      </animated.mesh>

      {/* Outer atmospheric glow */}
      <animated.mesh>
        <sphereGeometry args={[2.6, 32, 32]} />
        <animated.meshBasicMaterial
          color={color}
          transparent
          opacity={isClassifying ? 0.12 : result ? 0.1 : 0.04}
          toneMapped={false}
          depthWrite={false}
        />
      </animated.mesh>

      {/* Enhanced point light for self-illumination */}
      <pointLight
        position={[0, 0, 0]}
        intensity={isClassifying ? 2.0 : result ? 2.5 : 0.8}
        distance={12}
        decay={2}
        color={COLORS[currentState]}
        castShadow
      />
    </group>
  )
}
