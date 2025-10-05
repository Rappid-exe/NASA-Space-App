'use client'

import { useEffect, useState } from 'react'
import { EffectComposer, Bloom, DepthOfField, Vignette } from '@react-three/postprocessing'
import { BlendFunction } from 'postprocessing'
import { useThree } from '@react-three/fiber'

interface PostProcessingProps {
  enableBloom?: boolean
  enableDepthOfField?: boolean
  enableVignette?: boolean
  bloomIntensity?: number
  bloomLuminanceThreshold?: number
  bloomLuminanceSmoothing?: number
  dofFocusDistance?: number
  dofFocalLength?: number
  dofBokehScale?: number
  vignetteOffset?: number
  vignetteDarkness?: number
}

export default function PostProcessing({
  enableBloom = true,
  enableDepthOfField = false,
  enableVignette = true,
  bloomIntensity = 1.5,
  bloomLuminanceThreshold = 0.9,
  bloomLuminanceSmoothing = 0.9,
  dofFocusDistance = 0.02,
  dofFocalLength = 0.05,
  dofBokehScale = 2,
  vignetteOffset = 0.5,
  vignetteDarkness = 0.5,
}: PostProcessingProps) {
  const { gl, scene, camera } = useThree()
  const [isReady, setIsReady] = useState(false)

  useEffect(() => {
    // Ensure scene is ready before rendering effects
    if (gl && scene && camera) {
      setIsReady(true)
    }
  }, [gl, scene, camera])

  // Only render EffectComposer if at least one effect is enabled and scene is ready
  if (!isReady || (!enableBloom && !enableDepthOfField && !enableVignette)) {
    return null
  }

  // Render different combinations based on enabled effects
  if (enableBloom && enableDepthOfField && enableVignette) {
    return (
      <EffectComposer multisampling={0}>
        <Bloom
          intensity={bloomIntensity}
          luminanceThreshold={bloomLuminanceThreshold}
          luminanceSmoothing={bloomLuminanceSmoothing}
          mipmapBlur
          blendFunction={BlendFunction.ADD}
        />
        <DepthOfField
          focusDistance={dofFocusDistance}
          focalLength={dofFocalLength}
          bokehScale={dofBokehScale}
          height={480}
        />
        <Vignette
          offset={vignetteOffset}
          darkness={vignetteDarkness}
          blendFunction={BlendFunction.NORMAL}
        />
      </EffectComposer>
    )
  }

  if (enableBloom && enableVignette) {
    return (
      <EffectComposer multisampling={0}>
        <Bloom
          intensity={bloomIntensity}
          luminanceThreshold={bloomLuminanceThreshold}
          luminanceSmoothing={bloomLuminanceSmoothing}
          mipmapBlur
          blendFunction={BlendFunction.ADD}
        />
        <Vignette
          offset={vignetteOffset}
          darkness={vignetteDarkness}
          blendFunction={BlendFunction.NORMAL}
        />
      </EffectComposer>
    )
  }

  if (enableBloom && enableDepthOfField) {
    return (
      <EffectComposer multisampling={0}>
        <Bloom
          intensity={bloomIntensity}
          luminanceThreshold={bloomLuminanceThreshold}
          luminanceSmoothing={bloomLuminanceSmoothing}
          mipmapBlur
          blendFunction={BlendFunction.ADD}
        />
        <DepthOfField
          focusDistance={dofFocusDistance}
          focalLength={dofFocalLength}
          bokehScale={dofBokehScale}
          height={480}
        />
      </EffectComposer>
    )
  }

  if (enableDepthOfField && enableVignette) {
    return (
      <EffectComposer multisampling={0}>
        <DepthOfField
          focusDistance={dofFocusDistance}
          focalLength={dofFocalLength}
          bokehScale={dofBokehScale}
          height={480}
        />
        <Vignette
          offset={vignetteOffset}
          darkness={vignetteDarkness}
          blendFunction={BlendFunction.NORMAL}
        />
      </EffectComposer>
    )
  }

  if (enableBloom) {
    return (
      <EffectComposer multisampling={0}>
        <Bloom
          intensity={bloomIntensity}
          luminanceThreshold={bloomLuminanceThreshold}
          luminanceSmoothing={bloomLuminanceSmoothing}
          mipmapBlur
          blendFunction={BlendFunction.ADD}
        />
      </EffectComposer>
    )
  }

  if (enableDepthOfField) {
    return (
      <EffectComposer multisampling={0}>
        <DepthOfField
          focusDistance={dofFocusDistance}
          focalLength={dofFocalLength}
          bokehScale={dofBokehScale}
          height={480}
        />
      </EffectComposer>
    )
  }

  if (enableVignette) {
    return (
      <EffectComposer multisampling={0}>
        <Vignette
          offset={vignetteOffset}
          darkness={vignetteDarkness}
          blendFunction={BlendFunction.NORMAL}
        />
      </EffectComposer>
    )
  }

  return null
}
