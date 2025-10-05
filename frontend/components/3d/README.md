# 3D Components Documentation

## Classification Scene Components

### ClassificationScene
Main scene component for the classification page that displays an animated planet responding to classification states.

**Usage:**
```tsx
import ClassificationScene from '@/components/3d/scenes/ClassificationScene'

<ClassificationScene
  result={classificationResult?.prediction}
  isClassifying={loading}
  showParticles={true}
  className="h-screen"
/>
```

**Props:**
- `result`: `'CONFIRMED' | 'FALSE_POSITIVE' | null` - Classification result
- `isClassifying`: `boolean` - Whether classification is in progress
- `showParticles`: `boolean` - Show particle effects on successful classification (optional)
- `className`: `string` - Additional CSS classes

### ClassificationPlanet
Animated planet that changes color and behavior based on classification state.

**States:**
- **Idle** (no result): Grey planet with slow rotation
- **Classifying**: Grey planet with pulsing glow and faster rotation
- **CONFIRMED**: Blue/green planet with bright glow
- **FALSE_POSITIVE**: Red/orange planet with warm glow

**Features:**
- Smooth color transitions (500ms)
- Variable rotation speed based on state
- Emissive glow that pulses during classification
- Self-illuminating point light

### ZoomAnimation
Camera controller that creates a cinematic zoom-in effect on page load.

**Default behavior:**
- Starts at position `[0, 0, 20]`
- Zooms to `[0, 0, 8]` over 1.5 seconds
- Automatically starts on mount

### ParticleSystem
Optional particle effects that appear on successful classification.

**Features:**
- 100 particles (50 on mobile) that emit from planet surface
- Color matches classification result
- Particles fade and regenerate continuously
- Performance optimized with instancing

## Solar System Scene Components

### SolarSystemScene
Main scene for the homepage with orbiting planets and stars.

**Usage:**
```tsx
import SolarSystemScene from '@/components/3d/scenes/SolarSystemScene'

<SolarSystemScene className="h-screen" />
```

### Shared Components

#### Sun
Glowing sun with emissive material and point light.

#### Planet
Reusable planet component with orbital and self-rotation animations.

#### Starfield
Thousands of twinkling stars using instanced rendering for performance.

## Integration Example

To integrate the ClassificationScene into the classify page:

```tsx
'use client'

import { useState } from 'react'
import ClassificationScene from '@/components/3d/scenes/ClassificationScene'
import ClassificationForm from '@/components/ClassificationForm'
import ResultsDisplay from '@/components/ResultsDisplay'

export default function ClassifyPage() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  return (
    <div className="relative min-h-screen">
      {/* 3D Background */}
      <div className="fixed inset-0 z-0">
        <ClassificationScene
          result={result?.prediction}
          isClassifying={loading}
        />
      </div>

      {/* UI Overlay */}
      <div className="relative z-10">
        <ClassificationForm
          onResult={setResult}
          onLoadingChange={setLoading}
        />
        {result && <ResultsDisplay result={result} />}
      </div>
    </div>
  )
}
```

## Performance Considerations

- Mobile devices automatically get reduced quality (fewer stars, particles)
- Adaptive performance scaling built-in
- Instanced rendering for particles and stars
- Efficient animation loops using `useFrame`

## Browser Support

- Requires WebGL support
- Fallback handling should be implemented in parent components
- Tested on Chrome, Firefox, Safari, Edge
