# Frontend Setup Guide - Exoplanet Classifier

Complete guide to set up and run the Next.js frontend application.

## Quick Start

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Create environment file
cp .env.local.example .env.local

# Start development server
npm run dev
```

Open http://localhost:3000 in your browser.

## Prerequisites

### Required
- **Node.js**: Version 18.0.0 or higher
- **npm**: Version 9.0.0 or higher (comes with Node.js)
- **Backend API**: Running on http://localhost:8000

### Optional
- **yarn** or **pnpm**: Alternative package managers
- **Git**: For version control

## Step-by-Step Installation

### 1. Check Node.js Version

```bash
node --version
# Should show v18.0.0 or higher
```

If you need to install or update Node.js:
- Download from https://nodejs.org/
- Or use nvm: `nvm install 18`

### 2. Navigate to Frontend Directory

```bash
cd frontend
```

### 3. Install Dependencies

Using npm (recommended):
```bash
npm install
```

Using yarn:
```bash
yarn install
```

Using pnpm:
```bash
pnpm install
```

This will install:
- Next.js 14
- React 18
- TypeScript
- Tailwind CSS
- Recharts (for charts)
- Axios (for API calls)
- And all other dependencies

### 4. Configure Environment

Create `.env.local` file:
```bash
cp .env.local.example .env.local
```

Edit `.env.local`:
```env
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
```

**Important**: The `NEXT_PUBLIC_` prefix makes the variable available in the browser.

### 5. Start Backend API

Before starting the frontend, ensure your backend is running:

```bash
# In the project root directory (not frontend/)
python -m uvicorn api.main:app --reload
```

Verify it's running by visiting http://localhost:8000/docs

### 6. Start Development Server

```bash
npm run dev
```

You should see:
```
  ▲ Next.js 14.2.3
  - Local:        http://localhost:3000
  - Ready in 2.5s
```

### 7. Open in Browser

Navigate to http://localhost:3000

You should see the Exoplanet Classifier homepage!

## Verification Checklist

Use this checklist to verify everything is working:

- [ ] Node.js 18+ installed
- [ ] Dependencies installed successfully
- [ ] `.env.local` file created and configured
- [ ] Backend API running on port 8000
- [ ] Frontend running on port 3000
- [ ] Homepage loads without errors
- [ ] Dashboard shows model statistics (or warning if no model)
- [ ] Can navigate to /classify page
- [ ] Can navigate to /upload page
- [ ] No console errors in browser DevTools

## Common Issues & Solutions

### Issue: "Module not found" errors

**Solution**:
```bash
# Delete node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

### Issue: "Port 3000 already in use"

**Solution 1** - Kill the process:
```bash
# On Windows
netstat -ano | findstr :3000
taskkill /PID <PID> /F

# On Mac/Linux
lsof -ti:3000 | xargs kill -9
```

**Solution 2** - Use a different port:
```bash
npm run dev -- -p 3001
```

### Issue: "Unable to connect to API server"

**Checklist**:
1. Is the backend running? Check http://localhost:8000/health
2. Is `NEXT_PUBLIC_API_URL` correct in `.env.local`?
3. Are there CORS errors in browser console?
4. Is a firewall blocking the connection?

**Solution**:
```bash
# Restart backend with CORS enabled
python -m uvicorn api.main:app --reload --host 0.0.0.0
```

### Issue: TypeScript errors

**Solution**:
```bash
# Regenerate TypeScript types
npm run build
```

### Issue: Styling not working

**Solution**:
```bash
# Rebuild Tailwind CSS
npm run dev
# Or clear Next.js cache
rm -rf .next
npm run dev
```

### Issue: "Cannot find module '@/...'"

**Solution**: Check `tsconfig.json` has the correct path mapping:
```json
{
  "compilerOptions": {
    "paths": {
      "@/*": ["./*"]
    }
  }
}
```

## Development Workflow

### Hot Reload

Next.js automatically reloads when you save files:
- **Pages**: Instant reload
- **Components**: Instant reload
- **Styles**: Instant reload
- **Config files**: Requires restart

### File Structure

```
frontend/
├── app/                    # Pages (App Router)
│   ├── page.tsx           # Home page
│   ├── classify/          # Classification page
│   ├── upload/            # Upload page
│   └── layout.tsx         # Root layout
├── components/            # Reusable components
├── lib/                   # Utilities
│   ├── api.ts            # API client
│   └── types.ts          # TypeScript types
└── public/               # Static files
```

### Adding a New Page

1. Create file in `app/` directory:
```typescript
// app/new-page/page.tsx
export default function NewPage() {
  return <div>New Page</div>;
}
```

2. Access at http://localhost:3000/new-page

### Adding a New Component

1. Create file in `components/` directory:
```typescript
// components/MyComponent.tsx
export default function MyComponent() {
  return <div>My Component</div>;
}
```

2. Import and use:
```typescript
import MyComponent from '@/components/MyComponent';
```

## Building for Production

### Local Production Build

```bash
# Build the application
npm run build

# Start production server
npm start
```

### Analyze Bundle Size

```bash
npm run build
# Check .next/analyze/ for bundle analysis
```

### Optimize Images

Place images in `public/` directory:
```typescript
import Image from 'next/image';

<Image src="/logo.png" alt="Logo" width={100} height={100} />
```

## Testing

### Manual Testing Checklist

- [ ] Homepage loads and displays correctly
- [ ] Dashboard shows model statistics
- [ ] Classification form accepts input
- [ ] Classification returns results
- [ ] CSV upload works
- [ ] Batch results display correctly
- [ ] Navigation works between pages
- [ ] Responsive design works on mobile
- [ ] No console errors
- [ ] API errors are handled gracefully

### Browser Testing

Test on:
- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Mobile browsers (iOS Safari, Chrome Mobile)

## Deployment

### Deploy to Vercel (Recommended)

1. Push code to GitHub
2. Go to https://vercel.com
3. Import your repository
4. Set environment variable:
   - `NEXT_PUBLIC_API_URL`: Your production API URL
5. Deploy!

### Deploy to Netlify

```bash
npm run build
# Upload .next/ directory to Netlify
```

### Deploy with Docker

Create `Dockerfile`:
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

Build and run:
```bash
docker build -t exoplanet-frontend .
docker run -p 3000:3000 -e NEXT_PUBLIC_API_URL=http://api:8000 exoplanet-frontend
```

## Performance Optimization

### Tips for Better Performance

1. **Use Next.js Image component** for automatic optimization
2. **Lazy load components** with dynamic imports
3. **Minimize bundle size** by removing unused dependencies
4. **Enable caching** for API responses
5. **Use production build** for deployment

### Lighthouse Audit

Run Lighthouse in Chrome DevTools:
1. Open DevTools (F12)
2. Go to Lighthouse tab
3. Click "Generate report"
4. Aim for 90+ scores

## Environment Variables

### Available Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `NEXT_PUBLIC_API_URL` | Backend API URL | Yes | `http://localhost:8000` |

### Adding New Variables

1. Add to `.env.local`:
```env
NEXT_PUBLIC_MY_VAR=value
```

2. Use in code:
```typescript
const myVar = process.env.NEXT_PUBLIC_MY_VAR;
```

**Note**: Only variables prefixed with `NEXT_PUBLIC_` are available in the browser.

## Troubleshooting Commands

```bash
# Clear all caches and reinstall
rm -rf .next node_modules package-lock.json
npm install
npm run dev

# Check for outdated packages
npm outdated

# Update packages
npm update

# Audit for vulnerabilities
npm audit
npm audit fix

# Check TypeScript errors
npx tsc --noEmit

# Format code (if using Prettier)
npx prettier --write .
```

## Getting Help

If you encounter issues:

1. Check this guide first
2. Review the error message carefully
3. Check browser console for errors
4. Verify backend is running
5. Check `frontend/README.md` for more details
6. Search for the error online
7. Open an issue on GitHub

## Next Steps

After setup:

1. ✅ Verify all pages work
2. ✅ Test classification with example data
3. ✅ Upload a sample CSV file
4. ✅ Customize the UI if needed
5. ✅ Deploy to production

## Resources

- [Next.js Documentation](https://nextjs.org/docs)
- [React Documentation](https://react.dev)
- [Tailwind CSS Documentation](https://tailwindcss.com/docs)
- [TypeScript Documentation](https://www.typescriptlang.org/docs)
- [Recharts Documentation](https://recharts.org)

## Success!

If you've completed all steps and the application is running, congratulations! 🎉

You now have a fully functional exoplanet classification web interface.

Start classifying exoplanets at http://localhost:3000! 🚀🪐
