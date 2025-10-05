'use client'

import { useRef, useMemo } from 'react'
import { Points, BufferGeometry, PointsMaterial, BufferAttribute } from 'three'
import { useFrame } from '@react-three/fiber'

interface ParticleSystemProps {
  active?: boolean
  count?: number
  color?: string
  size?: number
  spread?: number
}

export default function ParticleSystem({
  active = false,
  count = 100,
  color = '#4A90E2',
  size = 0.05,
  spread = 3,
}: ParticleSystemProps) {
  const pointsRef = useRef<Points>(null)

  // Generate random particle positions
  const particles = useMemo(() => {
    const positions = new Float32Array(count * 3)
    const velocities = new Float32Array(count * 3)

    for (let i = 0; i < count; i++) {
      const i3 = i * 3
      // Start near the planet surface
      const theta = Math.random() * Math.PI * 2
      const phi = Math.random() * Math.PI
      const radius = 2.2 + Math.random() * 0.5

      positions[i3] = radius * Math.sin(phi) * Math.cos(theta)
      positions[i3 + 1] = radius * Math.sin(phi) * Math.sin(theta)
      positions[i3 + 2] = radius * Math.cos(phi)

      // Random outward velocities
      velocities[i3] = (Math.random() - 0.5) * 0.02
      velocities[i3 + 1] = (Math.random() - 0.5) * 0.02
      velocities[i3 + 2] = (Math.random() - 0.5) * 0.02
    }

    return { positions, velocities }
  }, [count, active]) // Regenerate when active changes

  // Animate particles
  useFrame((state, delta) => {
    if (!pointsRef.current || !active) return

    const positions = pointsRef.current.geometry.attributes.position
    const velocities = particles.velocities

    for (let i = 0; i < count; i++) {
      const i3 = i * 3

      // Update positions based on velocities
      positions.array[i3] += velocities[i3]
      positions.array[i3 + 1] += velocities[i3 + 1]
      positions.array[i3 + 2] += velocities[i3 + 2]

      // Fade out particles that are too far
      const distance = Math.sqrt(
        positions.array[i3] ** 2 +
        positions.array[i3 + 1] ** 2 +
        positions.array[i3 + 2] ** 2
      )

      if (distance > spread + 2) {
        // Reset particle to planet surface
        const theta = Math.random() * Math.PI * 2
        const phi = Math.random() * Math.PI
        const radius = 2.2

        positions.array[i3] = radius * Math.sin(phi) * Math.cos(theta)
        positions.array[i3 + 1] = radius * Math.sin(phi) * Math.sin(theta)
        positions.array[i3 + 2] = radius * Math.cos(phi)
      }
    }

    positions.needsUpdate = true
  })

  if (!active) return null

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={count}
          array={particles.positions}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        size={size}
        color={color}
        transparent
        opacity={0.8}
        sizeAttenuation
        depthWrite={false}
      />
    </points>
  )
}
