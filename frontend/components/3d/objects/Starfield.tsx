'use client'

import { useRef, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

interface StarfieldProps {
  count?: number
  radius?: number
  depth?: number
}

export default function Starfield({
  count = 5000,
  radius = 100,
  depth = 50,
}: StarfieldProps) {
  const pointsRef = useRef<THREE.Points>(null)

  // Generate random star positions and properties
  const [positions, colors, sizes, phases] = useMemo(() => {
    const positions = new Float32Array(count * 3)
    const colors = new Float32Array(count * 3)
    const sizes = new Float32Array(count)
    const phases = new Float32Array(count)

    for (let i = 0; i < count; i++) {
      // Random position in sphere
      const theta = Math.random() * Math.PI * 2
      const phi = Math.acos(Math.random() * 2 - 1)
      const r = radius + Math.random() * depth

      positions[i * 3] = r * Math.sin(phi) * Math.cos(theta)
      positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta)
      positions[i * 3 + 2] = r * Math.cos(phi)

      // Depth-based opacity (closer stars are brighter)
      const depthFactor = 1 - (r - radius) / depth
      const brightness = 0.5 + depthFactor * 0.5

      // Slight color variation (white to blue-white)
      colors[i * 3] = brightness
      colors[i * 3 + 1] = brightness
      colors[i * 3 + 2] = brightness * (0.9 + Math.random() * 0.1)

      // Random size based on depth
      sizes[i] = (0.5 + Math.random() * 1.5) * depthFactor

      // Random phase for twinkling
      phases[i] = Math.random() * Math.PI * 2
    }

    return [positions, colors, sizes, phases]
  }, [count, radius, depth])

  // Twinkling animation
  useFrame((state) => {
    if (pointsRef.current) {
      const geometry = pointsRef.current.geometry
      const sizeAttribute = geometry.attributes.size

      for (let i = 0; i < count; i++) {
        // Twinkling effect using sine wave with random phase
        const twinkle = Math.sin(state.clock.elapsedTime * 2 + phases[i]) * 0.3 + 0.7
        sizeAttribute.array[i] = sizes[i] * twinkle
      }

      sizeAttribute.needsUpdate = true
    }
  })

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={count}
          array={positions}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-color"
          count={count}
          array={colors}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-size"
          count={count}
          array={sizes}
          itemSize={1}
        />
      </bufferGeometry>
      <pointsMaterial
        size={1}
        sizeAttenuation
        vertexColors
        transparent
        opacity={0.8}
        depthWrite={false}
      />
    </points>
  )
}
