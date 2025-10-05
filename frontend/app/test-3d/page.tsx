'use client'

import dynamic from 'next/dynamic'

// Dynamically import the TestScene to avoid SSR issues with Three.js
const TestScene = dynamic(() => import('@/components/3d/TestScene'), {
  ssr: false,
  loading: () => (
    <div style={{ 
      width: '100%', 
      height: '100vh', 
      background: '#000', 
      display: 'flex', 
      alignItems: 'center', 
      justifyContent: 'center',
      color: 'white'
    }}>
      Loading 3D Scene...
    </div>
  )
})

export default function Test3DPage() {
  return <TestScene />
}
