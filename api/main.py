"""
FastAPI application for exoplanet classification inference.
Provides REST endpoints for real-time classification of astronomical observations.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.model_registry import ModelRegistry, ModelMetadata
from api.inference_service import InferenceService

# Initialize FastAPI app
app = FastAPI(
    title="Exoplanet Classification API",
    description="AI-powered exoplanet identification and classification service",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize inference service
inference_service = InferenceService()


# Pydantic models for request/response validation
class ExoplanetFeatures(BaseModel):
    """Input features for exoplanet classification."""
    orbital_period: float = Field(..., description="Orbital period in days", gt=0)
    transit_duration: float = Field(..., description="Transit duration in hours", gt=0)
    transit_depth: float = Field(..., description="Transit depth in ppm", gt=0)
    planetary_radius: float = Field(..., description="Planetary radius in Earth radii", gt=0)
    equilibrium_temperature: Optional[float] = Field(None, description="Equilibrium temperature in Kelvin", gt=0)
    
    @validator('orbital_period')
    def validate_period(cls, v):
        if v > 10000:  # Sanity check
            raise ValueError('Orbital period seems unreasonably large')
        return v
    
    @validator('transit_duration')
    def validate_duration(cls, v):
        if v > 100:  # Sanity check
            raise ValueError('Transit duration seems unreasonably large')
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "orbital_period": 3.52,
                "transit_duration": 2.8,
                "transit_depth": 500.0,
                "planetary_radius": 1.2,
                "equilibrium_temperature": 1200.0
            }]
        }
    }


class BatchClassificationRequest(BaseModel):
    """Request for batch classification of multiple observations."""
    observations: List[ExoplanetFeatures] = Field(..., description="List of observations to classify")
    
    @validator('observations')
    def validate_batch_size(cls, v):
        if len(v) > 1000:
            raise ValueError('Batch size cannot exceed 1000 observations')
        if len(v) == 0:
            raise ValueError('Batch must contain at least one observation')
        return v


class ClassificationResult(BaseModel):
    """Result of exoplanet classification."""
    prediction: str = Field(..., description="Predicted class (CONFIRMED or FALSE_POSITIVE)")
    confidence: float = Field(..., description="Confidence score (0-1)")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")
    explanation: str = Field(..., description="Human-readable explanation")


class BatchClassificationResponse(BaseModel):
    """Response for batch classification."""
    results: List[ClassificationResult]
    total_processed: int
    summary: Dict[str, int]


class ModelStatistics(BaseModel):
    """Model performance statistics."""
    model_id: str
    model_name: str
    algorithm: str
    version: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_date: str
    training_samples: int
    test_samples: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_id: Optional[str]


# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Exoplanet Classification API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    model_loaded = inference_service.model is not None
    model_id = inference_service.metadata.model_id if inference_service.metadata else None
    
    return HealthResponse(
        status="healthy" if model_loaded else "no_model_loaded",
        model_loaded=model_loaded,
        model_id=model_id
    )


@app.post("/classify", response_model=ClassificationResult, status_code=status.HTTP_200_OK)
async def classify_observation(features: ExoplanetFeatures):
    """
    Classify a single exoplanet observation.
    
    Args:
        features: Astronomical features of the observation
        
    Returns:
        Classification result with prediction and confidence
    """
    try:
        result = inference_service.classify_observation(features.model_dump())
        return ClassificationResult(**result)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                          detail=f"Classification failed: {str(e)}")


@app.post("/classify/batch", response_model=BatchClassificationResponse, status_code=status.HTTP_200_OK)
async def classify_batch(request: BatchClassificationRequest):
    """
    Classify multiple exoplanet observations in batch.
    
    Args:
        request: Batch classification request with list of observations
        
    Returns:
        Batch classification results with summary
    """
    try:
        observations = [obs.model_dump() for obs in request.observations]
        results = inference_service.classify_batch(observations)
        
        # Create summary
        summary = {"CONFIRMED": 0, "FALSE_POSITIVE": 0}
        for result in results:
            summary[result["prediction"]] = summary.get(result["prediction"], 0) + 1
        
        return BatchClassificationResponse(
            results=[ClassificationResult(**r) for r in results],
            total_processed=len(results),
            summary=summary
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                          detail=f"Batch classification failed: {str(e)}")


@app.get("/model/statistics", response_model=ModelStatistics)
async def get_model_statistics():
    """
    Get current model performance statistics.
    
    Returns:
        Model metadata and performance metrics
    """
    try:
        stats = inference_service.get_model_statistics()
        return ModelStatistics(**stats)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                          detail=f"Failed to retrieve model statistics: {str(e)}")


@app.get("/models/list", response_model=List[Dict[str, Any]])
async def list_available_models():
    """
    List all available models in the registry.
    
    Returns:
        List of available models with basic information
    """
    try:
        registry = ModelRegistry()
        models = registry.list_models()
        return models
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                          detail=f"Failed to list models: {str(e)}")


@app.post("/model/load/{model_id}")
async def load_model(model_id: str):
    """
    Load a specific model by ID.
    
    Args:
        model_id: Model identifier to load
        
    Returns:
        Success message with loaded model information
    """
    try:
        inference_service.load_model(model_id=model_id)
        return {
            "message": "Model loaded successfully",
            "model_id": model_id,
            "algorithm": inference_service.metadata.algorithm
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                          detail=f"Failed to load model: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
