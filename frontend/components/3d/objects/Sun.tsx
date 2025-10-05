'use client'

import { useRef } from 'react'
import { Mesh, PointLight } from 'three'
import { useFrame } from '@react-three/fiber'

interface SunProps {
  size?: number
  intensity?: number
}

export default function Sun({ size = 2, intensity = 2 }: SunProps) {
  const sunRef = useRef<Mesh>(null)
  const glowRef = useRef<Mesh>(null)
  const lightRef = useRef<PointLight>(null)

  // Refined pulsing animation with smoother easing
  useFrame((state) => {
    const time = state.clock.elapsedTime
    // Use multiple sine waves for more organic pulsing
    const pulse = Math.sin(time * 0.4) * 0.03 + Math.sin(time * 0.7) * 0.02 + 1
    
    if (sunRef.current) {
      sunRef.current.scale.setScalar(pulse)
    }
    if (glowRef.current) {
      glowRef.current.scale.setScalar(pulse * 1.15)
      // Subtle rotation for dynamic effect
      glowRef.current.rotation.z = time * 0.05
    }
    
    // Vary light intensity slightly for more realism
    if (lightRef.current) {
      lightRef.current.intensity = intensity * (0.95 + Math.sin(time * 0.6) * 0.05)
    }
  })

  return (
    <group>
      {/* Point light at center for illumination */}
      <pointLight
        ref={lightRef}
        position={[0, 0, 0]}
        intensity={intensity}
        distance={100}
        decay={2}
        color="#FDB813"
      />

      {/* Main sun sphere with enhanced emissive material */}
      <mesh ref={sunRef} castShadow>
        <sphereGeometry args={[size, 128, 128]} />
        <meshStandardMaterial
          emissive="#FFA500"
          emissiveIntensity={3.0}
          color="#FFD700"
          toneMapped={false}
          roughness={0.9}
          metalness={0.1}
        />
      </mesh>

      {/* Inner glow layer with gradient effect */}
      <mesh ref={glowRef}>
        <sphereGeometry args={[size * 1.25, 64, 64]} />
        <meshBasicMaterial
          color="#FFE55C"
          transparent
          opacity={0.4}
          toneMapped={false}
          depthWrite={false}
        />
      </mesh>

      {/* Middle glow layer */}
      <mesh>
        <sphereGeometry args={[size * 1.5, 48, 48]} />
        <meshBasicMaterial
          color="#FDB813"
          transparent
          opacity={0.2}
          toneMapped={false}
          depthWrite={false}
        />
      </mesh>

      {/* Outer glow effect for atmospheric halo */}
      <mesh>
        <sphereGeometry args={[size * 1.8, 32, 32]} />
        <meshBasicMaterial
          color="#FF8C00"
          transparent
          opacity={0.08}
          toneMapped={false}
          depthWrite={false}
        />
      </mesh>
    </group>
  )
}
