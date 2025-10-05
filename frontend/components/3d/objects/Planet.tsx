'use client'

import { useRef } from 'react'
import { Mesh, Group } from 'three'
import { useFrame } from '@react-three/fiber'
import { useTexture } from '@react-three/drei'

interface PlanetProps {
  name: string
  distance: number
  speed: number
  size: number
  color?: string
  texture?: string
  segments?: number
}

export default function Planet({
  name,
  distance,
  speed,
  size,
  color = '#888888',
  texture,
  segments = 64,
}: PlanetProps) {
  const orbitRef = useRef<Group>(null)
  const planetRef = useRef<Mesh>(null)

  // Load texture if provided (with error handling)
  let textureMap = null
  try {
    if (texture) {
      textureMap = useTexture(texture)
    }
  } catch (error) {
    console.warn(`Failed to load texture for ${name}:`, error)
  }

  // Refined orbital and self-rotation animation with smoother timing
  useFrame((state, delta) => {
    if (orbitRef.current) {
      // Orbital animation around sun with smooth easing
      orbitRef.current.rotation.y += speed * delta
    }
    if (planetRef.current) {
      // Self-rotation with varied speed based on planet size
      const rotationSpeed = 0.3 + (1 / size) * 0.2
      planetRef.current.rotation.y += delta * rotationSpeed
      // Subtle axial tilt for realism
      planetRef.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.1) * 0.05
    }
  })

  return (
    <group ref={orbitRef}>
      {/* Position planet at distance from center */}
      <mesh ref={planetRef} position={[distance, 0, 0]} castShadow receiveShadow>
        <sphereGeometry args={[size, segments, segments]} />
        {textureMap ? (
          <meshStandardMaterial 
            map={textureMap}
            roughness={0.8}
            metalness={0.2}
            envMapIntensity={0.5}
          />
        ) : (
          <meshStandardMaterial 
            color={color} 
            roughness={0.85} 
            metalness={0.15}
            emissive={color}
            emissiveIntensity={0.15}
            envMapIntensity={0.3}
          />
        )}
      </mesh>
      
      {/* Subtle atmospheric glow for larger planets */}
      {size > 0.8 && (
        <mesh position={[distance, 0, 0]}>
          <sphereGeometry args={[size * 1.08, 32, 32]} />
          <meshBasicMaterial
            color={color}
            transparent
            opacity={0.08}
            depthWrite={false}
          />
        </mesh>
      )}
    </group>
  )
}
