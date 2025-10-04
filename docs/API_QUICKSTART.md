# Exoplanet Classification API - Quick Start Guide

This guide will help you get started with the Exoplanet Classification API.

## Prerequisites

1. Ensure you have trained a model first:
   ```bash
   python test_model_training.py
   ```

2. Verify the API structure:
   ```bash
   python test_api_structure.py
   ```

## Starting the API Server

### Option 1: Development Mode (Recommended for testing)

```bash
python -m uvicorn api.main:app --reload
```

The `--reload` flag enables auto-reload when you make code changes.

### Option 2: Production Mode

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Option 3: Using the Python script directly

```bash
python api/main.py
```

## Accessing the API

Once the server is running, you can access:

- **API Base URL**: http://localhost:8000
- **Interactive Documentation (Swagger UI)**: http://localhost:8000/docs
- **Alternative Documentation (ReDoc)**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## Quick Examples

### 1. Check API Health

```bash
curl http://localhost:8000/health
```

### 2. Classify a Single Observation

```bash
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "orbital_period": 3.52,
    "transit_duration": 2.8,
    "transit_depth": 500.0,
    "planetary_radius": 1.2,
    "equilibrium_temperature": 1200.0
  }'
```

### 3. Batch Classification

```bash
curl -X POST "http://localhost:8000/classify/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "observations": [
      {
        "orbital_period": 3.52,
        "transit_duration": 2.8,
        "transit_depth": 500.0,
        "planetary_radius": 1.2,
        "equilibrium_temperature": 1200.0
      },
      {
        "orbital_period": 365.25,
        "transit_duration": 6.5,
        "transit_depth": 84.0,
        "planetary_radius": 1.0,
        "equilibrium_temperature": 288.0
      }
    ]
  }'
```

### 4. Get Model Statistics

```bash
curl http://localhost:8000/model/statistics
```

### 5. List Available Models

```bash
curl http://localhost:8000/models/list
```

## Using Python Requests

```python
import requests

# Base URL
BASE_URL = "http://localhost:8000"

# Classify a single observation
observation = {
    "orbital_period": 3.52,
    "transit_duration": 2.8,
    "transit_depth": 500.0,
    "planetary_radius": 1.2,
    "equilibrium_temperature": 1200.0
}

response = requests.post(f"{BASE_URL}/classify", json=observation)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Explanation: {result['explanation']}")
```

## Using JavaScript/Fetch

```javascript
const BASE_URL = 'http://localhost:8000';

async function classifyObservation() {
  const observation = {
    orbital_period: 3.52,
    transit_duration: 2.8,
    transit_depth: 500.0,
    planetary_radius: 1.2,
    equilibrium_temperature: 1200.0
  };

  const response = await fetch(`${BASE_URL}/classify`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(observation)
  });

  const result = await response.json();
  console.log('Prediction:', result.prediction);
  console.log('Confidence:', result.confidence);
}

classifyObservation();
```

## Interactive API Documentation

The easiest way to explore and test the API is through the interactive Swagger UI:

1. Start the API server
2. Open your browser to http://localhost:8000/docs
3. You'll see all available endpoints with:
   - Request/response schemas
   - Try it out functionality
   - Example values
   - Response codes

## Input Features Reference

| Feature | Type | Description | Unit | Required |
|---------|------|-------------|------|----------|
| orbital_period | float | Orbital period | days | Yes |
| transit_duration | float | Transit duration | hours | Yes |
| transit_depth | float | Transit depth | ppm | Yes |
| planetary_radius | float | Planetary radius | Earth radii | Yes |
| equilibrium_temperature | float | Equilibrium temperature | Kelvin | No |

## Response Format

### Single Classification Response

```json
{
  "prediction": "CONFIRMED",
  "confidence": 0.92,
  "probabilities": {
    "FALSE_POSITIVE": 0.08,
    "CONFIRMED": 0.92
  },
  "explanation": "This observation is classified as a CONFIRMED exoplanet with 92.0% confidence. The planetary radius suggests a Super-Earth. The orbital period is very short, indicating a hot planet close to its star."
}
```

### Batch Classification Response

```json
{
  "results": [
    {
      "prediction": "CONFIRMED",
      "confidence": 0.92,
      "probabilities": {
        "FALSE_POSITIVE": 0.08,
        "CONFIRMED": 0.92
      },
      "explanation": "..."
    }
  ],
  "total_processed": 1,
  "summary": {
    "CONFIRMED": 1,
    "FALSE_POSITIVE": 0
  }
}
```

## Troubleshooting

### "No module named 'fastapi'"

Install dependencies:
```bash
pip install -r requirements.txt
```

### "No model loaded"

Train a model first:
```bash
python test_model_training.py
```

### Port already in use

Use a different port:
```bash
python -m uvicorn api.main:app --port 8001
```

### CORS errors from browser

The API already has CORS enabled for all origins. If you still have issues, check your browser console for specific errors.

## Next Steps

- Explore the full API documentation at `/docs`
- Integrate the API with your web application
- Train additional models and compare performance
- Implement custom preprocessing for your specific use case

For more detailed information, see `api/README.md`.
