# 3D Scenes

This directory contains complete 3D scene compositions for different pages.

## SolarSystemScene

A complete solar system visualization with 8 planets orbiting a central sun.

### Features

- **8 Planets**: Mercury through Neptune with realistic colors and orbital speeds
- **Dynamic Sun**: Glowing sun with pulsing animation and point light
- **Starfield Background**: 5000 stars (2000 on mobile) with twinkling effect
- **Camera Drift**: Smooth camera movement using sine waves for dynamic feel
- **Lighting**: Ambient light + sun's point light for realistic illumination
- **Performance Optimizations**:
  - Adaptive quality based on device (mobile detection)
  - Reduced star count on mobile (2000 vs 5000)
  - Lower polygon count on mobile (32 vs 64 segments)
  - Adaptive DPR (device pixel ratio)
  - Performance monitoring with min threshold

### Usage

```tsx
import SolarSystemScene from '@/components/3d/scenes/SolarSystemScene'

export default function HomePage() {
  return (
    <div className="relative w-full h-screen">
      <SolarSystemScene className="absolute inset-0" />
      
      {/* Your UI overlay here */}
      <div className="absolute inset-0 pointer-events-none">
        <YourContent />
      </div>
    </div>
  )
}
```

### Configuration

Planet configurations can be modified in the `PLANET_CONFIGS` array:

```typescript
{
  name: 'earth',
  distance: 16,    // Distance from sun
  size: 1,         // Planet radius
  speed: 0.02,     // Orbital speed
  color: '#4A90E2' // Fallback color
}
```

### Performance

- **Desktop**: 5000 stars, 64 segments, full quality
- **Mobile**: 2000 stars, 32 segments, optimized quality
- **Adaptive DPR**: [1, 2] for balanced quality/performance
- **Performance threshold**: 0.5 (50% of target framerate)

### Testing

Visit `/solar-test` to see the scene in action with overlay UI.

## Requirements Satisfied

- ✅ 1.1: 3D solar system with planets orbiting central sun
- ✅ 1.3: Planets rotate smoothly at different speeds
- ✅ 1.4: Camera drifts slowly for dynamic feel
- ✅ 4.1: Performance optimizations (LOD, instancing, adaptive quality)
