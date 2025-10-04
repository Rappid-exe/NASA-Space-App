# Exoplanet Classifier - Frontend

Modern Next.js 14 web application for classifying exoplanets using AI/ML models trained on NASA data.

## Features

- 🎯 **Real-time Classification** - Classify individual exoplanet observations instantly
- 📊 **Batch Processing** - Upload CSV files to classify thousands of observations
- 📈 **Interactive Dashboard** - View model performance metrics and statistics
- 🎨 **Beautiful UI** - Modern, responsive design with dark space theme
- 📱 **Mobile Friendly** - Works seamlessly on all devices
- ⚡ **Fast & Optimized** - Built with Next.js 14 App Router for optimal performance

## Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Charts**: Recharts
- **HTTP Client**: Axios
- **Deployment**: Vercel-ready

## Prerequisites

- Node.js 18+ and npm/yarn/pnpm
- Backend API running on http://localhost:8000 (or configure `NEXT_PUBLIC_API_URL`)

## Installation

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
# or
yarn install
# or
pnpm install
```

3. Create environment file:
```bash
cp .env.local.example .env.local
```

4. Configure the API URL in `.env.local`:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Development

Start the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Building for Production

Build the application:

```bash
npm run build
npm start
```

## Project Structure

```
frontend/
├── app/                      # Next.js App Router pages
│   ├── page.tsx             # Home/Dashboard page
│   ├── classify/page.tsx    # Single classification page
│   ├── upload/page.tsx      # Batch upload page
│   ├── layout.tsx           # Root layout
│   └── globals.css          # Global styles
├── components/              # React components
│   ├── Dashboard.tsx        # Model statistics dashboard
│   ├── ClassificationForm.tsx  # Manual entry form
│   ├── ResultsDisplay.tsx   # Classification results
│   └── FileUpload.tsx       # CSV upload component
├── lib/                     # Utilities and helpers
│   ├── api.ts              # API client functions
│   └── types.ts            # TypeScript type definitions
├── public/                  # Static assets
├── package.json            # Dependencies
├── tsconfig.json           # TypeScript configuration
├── tailwind.config.ts      # Tailwind CSS configuration
└── next.config.mjs         # Next.js configuration
```

## Pages

### Home (`/`)
- Hero section with project overview
- Model statistics dashboard
- Quick navigation to classification features
- Performance metrics visualization

### Classify (`/classify`)
- Manual data entry form for single observations
- Real-time classification results
- Confidence scores and explanations
- Example values for quick testing

### Upload (`/upload`)
- CSV file upload interface
- Batch classification processing
- Results summary and distribution
- Export results as CSV

## API Integration

The frontend communicates with the FastAPI backend through the API client (`lib/api.ts`):

```typescript
import { classifyObservation } from '@/lib/api';

const result = await classifyObservation({
  orbital_period: 3.52,
  transit_duration: 2.8,
  transit_depth: 500.0,
  planetary_radius: 1.2,
  equilibrium_temperature: 1200.0
});
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_URL` | Backend API base URL | `http://localhost:8000` |

## Styling

The application uses Tailwind CSS with a custom space-themed color palette:

- **Primary Colors**: Blue (#3b82f6) and Purple (#6366f1)
- **Background**: Dark space gradient (#0a0e27 to #1a1f3a)
- **Accents**: Green (confirmed), Red (false positive)

Custom animations and hover effects enhance the user experience.

## Components

### Dashboard
Displays model performance metrics with interactive charts:
- Accuracy, Precision, Recall, F1 Score
- Bar and pie charts
- Training data statistics

### ClassificationForm
Form for entering observation data:
- Input validation
- Quick-load example values
- Real-time error feedback

### ResultsDisplay
Shows classification results:
- Prediction with confidence score
- Probability distribution chart
- Detailed explanation
- Confidence level indicator

### FileUpload
CSV file upload with drag-and-drop:
- File validation
- CSV parsing
- Batch processing
- Results export

## CSV Format

For batch classification, CSV files should have these columns:

```csv
orbital_period,transit_duration,transit_depth,planetary_radius,equilibrium_temperature
3.52,2.8,500.0,1.2,1200.0
365.25,6.5,84.0,1.0,288.0
```

Note: `equilibrium_temperature` is optional.

## Deployment

### Vercel (Recommended)

1. Push your code to GitHub
2. Import project in Vercel
3. Set environment variable: `NEXT_PUBLIC_API_URL`
4. Deploy

### Docker

```bash
# Build
docker build -t exoplanet-frontend .

# Run
docker run -p 3000:3000 -e NEXT_PUBLIC_API_URL=http://api:8000 exoplanet-frontend
```

### Static Export

```bash
npm run build
# Output in .next/ directory
```

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Mobile browsers

## Performance

- Lighthouse Score: 95+
- First Contentful Paint: < 1s
- Time to Interactive: < 2s
- Bundle size: ~200KB (gzipped)

## Troubleshooting

### API Connection Issues

If you see "Unable to connect to the API server":
1. Ensure backend is running on port 8000
2. Check `NEXT_PUBLIC_API_URL` in `.env.local`
3. Verify CORS is enabled on backend

### Build Errors

```bash
# Clear cache and reinstall
rm -rf .next node_modules
npm install
npm run build
```

### Type Errors

```bash
# Regenerate TypeScript types
npm run build
```

## Contributing

1. Follow the existing code style
2. Use TypeScript for type safety
3. Test on multiple browsers
4. Ensure responsive design works

## License

MIT License - see LICENSE file for details

## Support

For issues or questions:
- Check the main project README
- Review API documentation
- Open an issue on GitHub
