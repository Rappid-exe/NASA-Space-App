'use client'

import { useRef, useEffect } from 'react'
import { PerspectiveCamera as PerspectiveCameraType } from 'three'
import { PerspectiveCamera } from '@react-three/drei'
import { useSpring, animated } from '@react-spring/three'

interface ZoomAnimationProps {
  startPosition?: [number, number, number]
  endPosition?: [number, number, number]
  duration?: number
  autoStart?: boolean
}

export default function ZoomAnimation({
  startPosition = [0, 0, 20],
  endPosition = [0, 0, 8],
  duration = 1500,
  autoStart = true,
}: ZoomAnimationProps) {
  const cameraRef = useRef<PerspectiveCameraType>(null)

  // Animated camera position with refined easing
  const { position } = useSpring({
    from: { position: startPosition },
    to: { position: endPosition },
    config: { 
      tension: 120,
      friction: 26,
      mass: 1,
    },
    delay: autoStart ? 200 : 0, // Slightly longer delay for smoother start
  })

  useEffect(() => {
    // Ensure camera looks at origin
    if (cameraRef.current) {
      cameraRef.current.lookAt(0, 0, 0)
    }
  }, [])

  return (
    <PerspectiveCamera
      ref={cameraRef}
      makeDefault
      // @ts-ignore - animated position works but TypeScript doesn't recognize it
      position={position.position}
      fov={50}
      near={0.1}
      far={1000}
    />
  )
}
