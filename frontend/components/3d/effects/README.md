# 3D Effects Components

This directory contains visual effects and enhancements for the 3D scenes.

## Components

### PostProcessing.tsx

Provides cinematic post-processing effects using `@react-three/postprocessing`.

**Features:**
- **Bloom Effect**: Creates glowing halos around bright objects (sun, planets)
- **Depth of Field**: Adds cinematic focus blur for foreground/background
- **Vignette**: Darkens edges for depth and focus

**Usage:**
```tsx
<PostProcessing
  enableBloom={true}
  enableVignette={true}
  enableDepthOfField={false}
  bloomIntensity={1.5}
  bloomLuminanceThreshold={0.9}
  vignetteOffset={0.5}
  vignetteDarkness={0.5}
/>
```

**Props:**
- `enableBloom` (boolean): Enable bloom effect
- `enableDepthOfField` (boolean): Enable depth of field
- `enableVignette` (boolean): Enable vignette effect
- `bloomIntensity` (number): Intensity of bloom glow (default: 1.5)
- `bloomLuminanceThreshold` (number): Brightness threshold for bloom (default: 0.9)
- `bloomLuminanceSmoothing` (number): Smoothness of bloom transition (default: 0.9)
- `dofFocusDistance` (number): Focus distance for DOF (default: 0.02)
- `dofFocalLength` (number): Focal length for DOF (default: 0.05)
- `dofBokehScale` (number): Bokeh blur size (default: 2)
- `vignetteOffset` (number): Vignette start position (default: 0.5)
- `vignetteDarkness` (number): Vignette darkness intensity (default: 0.5)

### LoadingScreen.tsx

Displays a polished loading screen with progress indicator while 3D assets load.

**Features:**
- Animated spinner with glow effect
- Progress bar with percentage
- Asset loading status
- Error display
- Smooth fade-out transition

**Usage:**
```tsx
<LoadingScreen onLoadComplete={() => setSceneLoaded(true)} />
```

**Props:**
- `onLoadComplete` (function): Callback when loading completes

### ParticleSystem.tsx

Creates particle effects for classification results (already implemented in previous tasks).

## Performance Considerations

- Post-processing effects are disabled on mobile devices for better performance
- Effects are only rendered when at least one is enabled
- Bloom uses mipmap blur for better performance
- Shadow rendering is disabled on mobile

## Visual Quality Settings

### Desktop (High Quality)
- All post-processing effects enabled
- Shadows enabled
- High segment counts for geometry
- Full particle counts

### Mobile (Optimized)
- Post-processing disabled
- Shadows disabled
- Reduced segment counts
- Reduced particle counts

## Dependencies

- `@react-three/postprocessing` - Post-processing effects
- `@react-three/drei` - useProgress hook for loading
- `postprocessing` - Core post-processing library
