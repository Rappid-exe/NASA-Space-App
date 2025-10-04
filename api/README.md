# Exoplanet Classification API

REST API for real-time exoplanet classification using trained machine learning models.

## Features

- **Single Observation Classification**: Classify individual exoplanet candidates
- **Batch Classification**: Process multiple observations efficiently
- **Model Statistics**: Access current model performance metrics
- **Input Validation**: Automatic validation of astronomical feature data
- **Interactive Documentation**: Built-in Swagger UI for API exploration

## Installation

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

## Starting the API Server

### Development Mode (with auto-reload)

```bash
python -m uvicorn api.main:app --reload
```

### Production Mode

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API Base**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## API Endpoints

### Health Check

**GET** `/health`

Check API health and model status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_id": "exoplanet_classifier_v1_20250104_120000"
}
```

### Classify Single Observation

**POST** `/classify`

Classify a single exoplanet observation.

**Request Body:**
```json
{
  "orbital_period": 3.52,
  "transit_duration": 2.8,
  "transit_depth": 500.0,
  "planetary_radius": 1.2,
  "equilibrium_temperature": 1200.0
}
```

**Response:**
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

### Batch Classification

**POST** `/classify/batch`

Classify multiple observations in a single request.

**Request Body:**
```json
{
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
}
```

**Response:**
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
    },
    {
      "prediction": "CONFIRMED",
      "confidence": 0.88,
      "probabilities": {
        "FALSE_POSITIVE": 0.12,
        "CONFIRMED": 0.88
      },
      "explanation": "..."
    }
  ],
  "total_processed": 2,
  "summary": {
    "CONFIRMED": 2,
    "FALSE_POSITIVE": 0
  }
}
```

### Model Statistics

**GET** `/model/statistics`

Get current model performance metrics.

**Response:**
```json
{
  "model_id": "exoplanet_classifier_v1_20250104_120000",
  "model_name": "exoplanet_classifier",
  "algorithm": "RandomForest",
  "version": 1,
  "accuracy": 0.9234,
  "precision": 0.9156,
  "recall": 0.9312,
  "f1_score": 0.9233,
  "training_date": "2025-01-04T12:00:00",
  "training_samples": 8000,
  "test_samples": 2000
}
```

### List Available Models

**GET** `/models/list`

List all models in the registry.

**Response:**
```json
[
  {
    "model_id": "exoplanet_classifier_v1_20250104_120000",
    "version": 1,
    "algorithm": "RandomForest",
    "f1_score": 0.9233,
    "accuracy": 0.9234,
    "training_date": "2025-01-04T12:00:00"
  }
]
```

### Load Specific Model

**POST** `/model/load/{model_id}`

Load a specific model by ID.

**Response:**
```json
{
  "message": "Model loaded successfully",
  "model_id": "exoplanet_classifier_v1_20250104_120000",
  "algorithm": "RandomForest"
}
```

## Input Features

All classification endpoints require the following features:

| Feature | Type | Description | Unit | Required |
|---------|------|-------------|------|----------|
| `orbital_period` | float | Orbital period | days | Yes |
| `transit_duration` | float | Transit duration | hours | Yes |
| `transit_depth` | float | Transit depth | ppm | Yes |
| `planetary_radius` | float | Planetary radius | Earth radii | Yes |
| `equilibrium_temperature` | float | Equilibrium temperature | Kelvin | No |

### Validation Rules

- All values must be positive (> 0)
- `orbital_period` must be < 10,000 days
- `transit_duration` must be < 100 hours
- Batch requests limited to 1,000 observations

## Testing

Run the test suite to verify API functionality:

```bash
python test_inference_api.py
```

## Example Usage

### Python

```python
import requests

# Single observation
response = requests.post(
    "http://localhost:8000/classify",
    json={
        "orbital_period": 3.52,
        "transit_duration": 2.8,
        "transit_depth": 500.0,
        "planetary_radius": 1.2,
        "equilibrium_temperature": 1200.0
    }
)
result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### cURL

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

### JavaScript

```javascript
fetch('http://localhost:8000/classify', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    orbital_period: 3.52,
    transit_duration: 2.8,
    transit_depth: 500.0,
    planetary_radius: 1.2,
    equilibrium_temperature: 1200.0
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

## Error Handling

The API returns standard HTTP status codes:

- **200**: Success
- **400**: Bad Request (invalid input)
- **500**: Internal Server Error

Error responses include a `detail` field with the error message:

```json
{
  "detail": "Orbital period seems unreasonably large"
}
```

## Architecture

The API consists of two main components:

1. **FastAPI Application** (`api/main.py`): Handles HTTP requests, input validation, and response formatting
2. **Inference Service** (`api/inference_service.py`): Manages model loading, feature preparation, and predictions

The service automatically:
- Loads the best performing model from the registry
- Creates derived features from input data
- Generates human-readable explanations
- Handles batch processing efficiently

## Requirements

See `requirements.txt` for full dependencies. Key packages:

- `fastapi>=0.109.0`: Web framework
- `uvicorn>=0.27.0`: ASGI server
- `pydantic>=2.5.3`: Data validation
- `scikit-learn>=1.3.2`: ML models
- `pandas>=2.1.4`: Data processing
- `numpy>=1.26.2`: Numerical operations
