# Frontend Quick Reference

Quick commands and tips for working with the Exoplanet Classifier frontend.

## 🚀 Quick Start

```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

Open http://localhost:3000

## 📦 NPM Commands

```bash
npm run dev          # Start development server
npm run build        # Build for production
npm start            # Start production server
npm run lint         # Run ESLint
```

## 🌐 URLs

| Page | URL | Description |
|------|-----|-------------|
| Home | http://localhost:3000 | Dashboard and overview |
| Classify | http://localhost:3000/classify | Single observation |
| Upload | http://localhost:3000/upload | Batch CSV upload |
| API Docs | http://localhost:8000/docs | Backend API docs |

## 📁 File Structure

```
frontend/
├── app/                    # Pages (Next.js App Router)
│   ├── page.tsx           # Homepage
│   ├── classify/page.tsx  # Classification page
│   ├── upload/page.tsx    # Upload page
│   ├── layout.tsx         # Root layout
│   └── globals.css        # Global styles
├── components/            # React components
│   ├── Dashboard.tsx
│   ├── ClassificationForm.tsx
│   ├── ResultsDisplay.tsx
│   └── FileUpload.tsx
├── lib/                   # Utilities
│   ├── api.ts            # API client
│   └── types.ts          # TypeScript types
└── public/               # Static files
```

## 🎨 Color Palette

```css
/* Space Theme */
--space-darker: #0a0e27
--space-dark: #1a1f3a
--primary-blue: #3b82f6
--primary-purple: #6366f1
--success-green: #10b981
--error-red: #ef4444
```

## 🔧 Environment Variables

```env
# .env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 📊 API Client Usage

```typescript
import { classifyObservation, classifyBatch, getModelStatistics } from '@/lib/api';

// Single classification
const result = await classifyObservation({
  orbital_period: 3.52,
  transit_duration: 2.8,
  transit_depth: 500.0,
  planetary_radius: 1.2,
  equilibrium_temperature: 1200.0
});

// Batch classification
const results = await classifyBatch({
  observations: [/* array of observations */]
});

// Get model stats
const stats = await getModelStatistics();
```

## 🎯 TypeScript Types

```typescript
// Observation input
interface ExoplanetFeatures {
  orbital_period: number;
  transit_duration: number;
  transit_depth: number;
  planetary_radius: number;
  equilibrium_temperature?: number;
}

// Classification result
interface ClassificationResult {
  prediction: 'CONFIRMED' | 'FALSE_POSITIVE';
  confidence: number;
  probabilities: {
    FALSE_POSITIVE: number;
    CONFIRMED: number;
  };
  explanation: string;
}
```

## 🐛 Common Issues

### Port 3000 in use
```bash
# Kill process
lsof -ti:3000 | xargs kill -9

# Or use different port
npm run dev -- -p 3001
```

### API connection failed
```bash
# Check backend is running
curl http://localhost:8000/health

# Check environment variable
cat .env.local
```

### Module not found
```bash
# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

### Build errors
```bash
# Clear cache
rm -rf .next
npm run build
```

## 📝 CSV Format

```csv
orbital_period,transit_duration,transit_depth,planetary_radius,equilibrium_temperature
3.52,2.8,500.0,1.2,1200.0
365.25,6.5,84.0,1.0,288.0
```

## 🎨 Tailwind Classes

```tsx
// Common patterns
<div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
  {/* Card */}
</div>

<button className="px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all">
  {/* Gradient button */}
</button>

<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
  {/* Responsive grid */}
</div>
```

## 🔍 Debugging

```typescript
// Enable verbose logging
console.log('API Response:', response);

// Check environment
console.log('API URL:', process.env.NEXT_PUBLIC_API_URL);

// Network tab in DevTools
// Check for failed requests
```

## 📱 Responsive Breakpoints

```css
/* Tailwind breakpoints */
sm: 640px   /* Tablet */
md: 768px   /* Desktop */
lg: 1024px  /* Large desktop */
xl: 1280px  /* Extra large */
```

## 🚢 Deployment

### Vercel
```bash
# Push to GitHub, then:
# 1. Import project in Vercel
# 2. Set NEXT_PUBLIC_API_URL
# 3. Deploy
```

### Docker
```bash
docker build -t exoplanet-frontend .
docker run -p 3000:3000 -e NEXT_PUBLIC_API_URL=http://api:8000 exoplanet-frontend
```

## 🧪 Testing Checklist

- [ ] Homepage loads
- [ ] Dashboard shows stats
- [ ] Classification works
- [ ] CSV upload works
- [ ] Mobile responsive
- [ ] No console errors
- [ ] API errors handled

## 📚 Resources

- [Next.js Docs](https://nextjs.org/docs)
- [Tailwind Docs](https://tailwindcss.com/docs)
- [TypeScript Docs](https://www.typescriptlang.org/docs)
- [Recharts Docs](https://recharts.org)

## 💡 Tips

1. **Use TypeScript** - Catch errors early
2. **Check DevTools** - Network and Console tabs
3. **Test on mobile** - Use responsive mode
4. **Clear cache** - When things break
5. **Read error messages** - They're usually helpful

## 🎓 Example Values

### Hot Jupiter (Confirmed)
```
Period: 3.52 days
Duration: 2.8 hours
Depth: 500 ppm
Radius: 1.2 Earth radii
Temp: 1200 K
```

### Earth-like (Confirmed)
```
Period: 365.25 days
Duration: 6.5 hours
Depth: 84 ppm
Radius: 1.0 Earth radii
Temp: 288 K
```

### False Positive
```
Period: 0.5 days
Duration: 1.2 hours
Depth: 2000 ppm
Radius: 0.8 Earth radii
Temp: 2500 K
```

## 🆘 Getting Help

1. Check this reference
2. Read error messages
3. Check browser console
4. Review documentation
5. Search online
6. Ask for help

---

**Happy coding! 🚀🪐**
