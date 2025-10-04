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


@app.get("/model/feature-importance")
async def get_feature_importance():
    """
    Get feature importance scores for the current model.
    
    Returns:
        Dictionary with feature names and importance scores
    """
    try:
        importance = inference_service.get_feature_importance()
        return importance
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                          detail=f"Failed to get feature importance: {str(e)}")


@app.get("/education/exoplanet-info")
async def get_exoplanet_education():
    """
    Get educational information about exoplanets and their characteristics.
    
    Returns:
        Educational content about exoplanets
    """
    return {
        "overview": "Exoplanets are planets that orbit stars outside our solar system. They are detected using various methods, with the transit method being one of the most successful.",
        "detection_methods": {
            "transit": {
                "name": "Transit Method",
                "description": "Detects planets by measuring the dimming of a star's light as a planet passes in front of it",
                "key_features": ["orbital_period", "transit_duration", "transit_depth"]
            }
        },
        "planet_types": {
            "earth_like": {
                "name": "Earth-like",
                "radius_range": "< 1.5 Earth radii",
                "description": "Rocky planets similar in size to Earth",
                "habitability": "Potentially habitable if in the right zone"
            },
            "super_earth": {
                "name": "Super-Earth",
                "radius_range": "1.5 - 2.5 Earth radii",
                "description": "Larger rocky planets, more massive than Earth",
                "habitability": "May be habitable depending on atmosphere"
            },
            "neptune_like": {
                "name": "Neptune-like",
                "radius_range": "2.5 - 6 Earth radii",
                "description": "Gas planets similar to Neptune",
                "habitability": "Unlikely to be habitable"
            },
            "jupiter_like": {
                "name": "Jupiter-like",
                "radius_range": "> 6 Earth radii",
                "description": "Large gas giants",
                "habitability": "Not habitable"
            }
        },
        "features_explained": {
            "orbital_period": "Time it takes for the planet to complete one orbit around its star (in days)",
            "transit_duration": "How long the planet takes to cross in front of its star (in hours)",
            "transit_depth": "How much the star's brightness decreases during transit (in parts per million)",
            "planetary_radius": "Size of the planet compared to Earth",
            "equilibrium_temperature": "Estimated temperature of the planet based on stellar radiation (in Kelvin)"
        },
        "missions": {
            "kepler": {
                "name": "Kepler Space Telescope",
                "years": "2009-2018",
                "discoveries": "Over 2,600 confirmed exoplanets",
                "focus": "Continuous monitoring of 150,000 stars"
            },
            "tess": {
                "name": "Transiting Exoplanet Survey Satellite (TESS)",
                "years": "2018-present",
                "discoveries": "Over 5,000 candidate exoplanets",
                "focus": "All-sky survey of nearby bright stars"
            },
            "k2": {
                "name": "K2 Mission",
                "years": "2014-2018",
                "discoveries": "Over 500 confirmed exoplanets",
                "focus": "Extended Kepler mission with different fields"
            }
        }
    }


@app.get("/datasets/comparison")
async def get_dataset_comparison():
    """
    Get comparison statistics between different NASA mission datasets.
    
    Returns:
        Comparison data for Kepler, TESS, and K2 missions
    """
    try:
        # Import data loader
        from data.dataset_loader import DatasetLoader
        from pathlib import Path
        
        loader = DatasetLoader()
        data_dir = Path(__file__).parent.parent / 'data' / 'raw'
        
        comparison = {
            "missions": [],
            "summary": {
                "total_observations": 0,
                "total_confirmed": 0,
                "total_false_positives": 0
            }
        }
        
        # Try to load each dataset
        datasets = [
            ("Kepler", "cumulative.csv", "koi_disposition"),
            ("K2", "k2targets.csv", "k2_disposition"),
            ("TESS", "toi.csv", "tfopwg_disp")
        ]
        
        for mission_name, filename, disp_col in datasets:
            try:
                filepath = data_dir / filename
                if filepath.exists():
                    df = loader.load_dataset(str(filepath))
                    
                    # Count dispositions
                    confirmed = 0
                    false_positive = 0
                    candidate = 0
                    
                    if disp_col in df.columns:
                        disp_counts = df[disp_col].value_counts()
                        confirmed = disp_counts.get('CONFIRMED', 0) + disp_counts.get('CP', 0)
                        false_positive = disp_counts.get('FALSE POSITIVE', 0) + disp_counts.get('FP', 0)
                        candidate = disp_counts.get('CANDIDATE', 0) + disp_counts.get('PC', 0)
                    
                    mission_data = {
                        "name": mission_name,
                        "total_observations": len(df),
                        "confirmed_exoplanets": int(confirmed),
                        "false_positives": int(false_positive),
                        "candidates": int(candidate),
                        "confirmation_rate": float(confirmed / len(df) * 100) if len(df) > 0 else 0
                    }
                    
                    comparison["missions"].append(mission_data)
                    comparison["summary"]["total_observations"] += len(df)
                    comparison["summary"]["total_confirmed"] += int(confirmed)
                    comparison["summary"]["total_false_positives"] += int(false_positive)
            except Exception as e:
                print(f"Could not load {mission_name} dataset: {e}")
        
        return comparison
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                          detail=f"Failed to generate dataset comparison: {str(e)}")


class HyperparameterTuningRequest(BaseModel):
    """Request for hyperparameter tuning."""
    algorithm: str = Field(..., description="Algorithm to tune (RandomForest, NeuralNetwork, SVM)")
    param_grid: Dict[str, List[Any]] = Field(..., description="Parameter grid for tuning")
    cv_folds: int = Field(default=5, description="Number of cross-validation folds", ge=2, le=10)


@app.post("/model/tune-hyperparameters")
async def tune_hyperparameters(request: HyperparameterTuningRequest):
    """
    Tune hyperparameters for a specific algorithm.
    
    Args:
        request: Hyperparameter tuning configuration
        
    Returns:
        Best parameters and performance metrics
    """
    try:
        from models.model_trainer import ModelTrainer
        from data.dataset_loader import DatasetLoader
        from data.data_processor import DataProcessor
        from pathlib import Path
        from sklearn.model_selection import GridSearchCV
        import numpy as np
        
        # Load and prepare data
        loader = DatasetLoader()
        processor = DataProcessor()
        data_dir = Path(__file__).parent.parent / 'data' / 'raw'
        
        # Try to load Kepler data
        kepler_path = data_dir / 'cumulative.csv'
        if not kepler_path.exists():
            raise ValueError("Training data not found")
        
        df = loader.load_dataset(str(kepler_path))
        X, y, feature_names = processor.prepare_features(df, target_column='koi_disposition')
        
        # Create base model
        if request.algorithm == 'RandomForest':
            from models.random_forest_classifier import RandomForestClassifier
            base_model = RandomForestClassifier()
        elif request.algorithm == 'NeuralNetwork':
            raise ValueError("Neural Network hyperparameter tuning not supported via this endpoint")
        elif request.algorithm == 'SVM':
            from models.svm_classifier import SVMClassifier
            base_model = SVMClassifier()
        else:
            raise ValueError(f"Unknown algorithm: {request.algorithm}")
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model.model,
            request.param_grid,
            cv=request.cv_folds,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        return {
            "best_params": grid_search.best_params_,
            "best_score": float(grid_search.best_score_),
            "cv_results": {
                "mean_scores": grid_search.cv_results_['mean_test_score'].tolist()[:10],  # Top 10
                "std_scores": grid_search.cv_results_['std_test_score'].tolist()[:10]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                          detail=f"Hyperparameter tuning failed: {str(e)}")


class RetrainingRequest(BaseModel):
    """Request for model retraining."""
    algorithm: str = Field(..., description="Algorithm to train")
    dataset: str = Field(default="kepler", description="Dataset to use (kepler, k2, tess, combined)")
    hyperparameters: Optional[Dict[str, Any]] = Field(None, description="Custom hyperparameters")


@app.post("/model/retrain")
async def retrain_model(request: RetrainingRequest):
    """
    Retrain a model with specified configuration.
    
    Args:
        request: Retraining configuration
        
    Returns:
        Training results and new model information
    """
    try:
        from models.model_trainer import ModelTrainer
        from models.model_registry import ModelRegistry
        from data.dataset_loader import DatasetLoader
        from data.data_processor import DataProcessor
        from pathlib import Path
        import numpy as np
        
        # Load data
        loader = DatasetLoader()
        processor = DataProcessor()
        data_dir = Path(__file__).parent.parent / 'data' / 'raw'
        
        # Select dataset
        if request.dataset == "kepler":
            filepath = data_dir / 'cumulative.csv'
            target_col = 'koi_disposition'
        elif request.dataset == "k2":
            filepath = data_dir / 'k2targets.csv'
            target_col = 'k2_disposition'
        elif request.dataset == "tess":
            filepath = data_dir / 'toi.csv'
            target_col = 'tfopwg_disp'
        else:
            raise ValueError(f"Unknown dataset: {request.dataset}")
        
        if not filepath.exists():
            raise ValueError(f"Dataset not found: {filepath}")
        
        df = loader.load_dataset(str(filepath))
        X, y, feature_names = processor.prepare_features(df, target_column=target_col)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
        
        # Configure model
        model_config = request.hyperparameters or {}
        if request.algorithm == 'RandomForest':
            model_config.setdefault('n_estimators', 100)
            model_config.setdefault('random_state', 42)
        elif request.algorithm == 'NeuralNetwork':
            model_config.setdefault('hidden_layers', (128, 64, 32))
            model_config.setdefault('random_state', 42)
        elif request.algorithm == 'SVM':
            model_config.setdefault('kernel', 'rbf')
            model_config.setdefault('random_state', 42)
        
        # Train model
        trainer = ModelTrainer()
        trainer.initialize_models({request.algorithm: model_config})
        trainer.train_models(X_train, y_train, X_val, y_val, feature_names)
        results = trainer.evaluate_models(X_test, y_test)
        
        # Register model
        registry = ModelRegistry()
        model = trainer.models[request.algorithm]
        evaluator = trainer.evaluator
        
        model_name = f"{request.dataset}_{request.algorithm.lower()}"
        model_id = registry.register_model(
            model=model,
            model_name=model_name,
            algorithm=request.algorithm,
            training_dataset=request.dataset,
            feature_columns=feature_names,
            evaluation_results=evaluator.evaluation_results[request.algorithm],
            training_samples=len(X_train),
            test_samples=len(X_test)
        )
        
        # Load the new model
        inference_service.load_model(model_id=model_id)
        
        return {
            "message": "Model retrained successfully",
            "model_id": model_id,
            "algorithm": request.algorithm,
            "dataset": request.dataset,
            "performance": results[request.algorithm]
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                          detail=f"Model retraining failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
