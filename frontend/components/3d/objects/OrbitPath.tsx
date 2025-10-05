'use client'

import { useMemo } from 'react'
import * as THREE from 'three'

interface OrbitPathProps {
  radius: number
  color?: string
  opacity?: number
}

export default function OrbitPath({ 
  radius, 
  color = '#ffffff', 
  opacity = 0.15 
}: OrbitPathProps) {
  // Create orbit ring geometry
  const geometry = useMemo(() => {
    const curve = new THREE.EllipseCurve(
      0, 0,           // center x, y
      radius, radius, // xRadius, yRadius
      0, 2 * Math.PI, // start angle, end angle
      false,          // clockwise
      0               // rotation
    )
    
    const points = curve.getPoints(128)
    const geometry = new THREE.BufferGeometry().setFromPoints(points)
    
    return geometry
  }, [radius])

  return (
    <line geometry={geometry} rotation={[-Math.PI / 2, 0, 0]}>
      <lineBasicMaterial 
        color={color} 
        transparent 
        opacity={opacity}
        linewidth={1}
      />
    </line>
  )
}
