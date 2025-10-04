# Advanced Features Documentation

This document describes the advanced features implemented for the hackathon demonstration.

## Overview

The advanced features provide powerful tools for model optimization, analysis, and education about exoplanet discovery. These features are accessible through the `/advanced` page in the web interface.

## Features

### 1. Feature Importance Visualization

**Purpose**: Understand which astronomical features contribute most to the model's predictions.

**Capabilities**:
- Visual representation of feature importance scores
- Top 5 most important features highlighted
- Explanations of what each feature means
- Algorithm-specific importance calculations

**API Endpoint**: `GET /model/feature-importance`

**How to Use**:
1. Navigate to Advanced Features → Feature Importance
2. View the importance scores for all features
3. Understand which features drive the model's decisions

**Supported Algorithms**:
- Random Forest: Uses built-in feature importances
- SVM: Uses absolute coefficient values
- Neural Network: Returns uniform importance (no built-in feature importance)

### 2. Dataset Comparison

**Purpose**: Compare exoplanet discoveries across different NASA missions.

**Capabilities**:
- Side-by-side comparison of Kepler, TESS, and K2 datasets
- Statistics on confirmed exoplanets, false positives, and candidates
- Confirmation rate analysis
- Visual charts showing distribution across missions
- Educational information about each mission

**API Endpoint**: `GET /datasets/comparison`

**How to Use**:
1. Navigate to Advanced Features → Dataset Comparison
2. View statistics for each mission
3. Compare discovery rates and dataset characteristics
4. Learn about the different NASA missions

**Requirements**:
- Dataset files must be present in `data/raw/` directory
- Supported datasets: `cumulative.csv` (Kepler), `k2targets.csv` (K2), `toi.csv` (TESS)

### 3. Hyperparameter Tuning

**Purpose**: Optimize model performance by finding the best hyperparameter configuration.

**Capabilities**:
- Grid search with cross-validation
- Configurable CV folds (2-10)
- Predefined parameter grids for each algorithm
- Performance visualization across different configurations
- Best parameter recommendations

**API Endpoint**: `POST /model/tune-hyperparameters`

**How to Use**:
1. Navigate to Advanced Features → Hyperparameter Tuning
2. Select the algorithm to tune (Random Forest or SVM)
3. Configure cross-validation folds
4. Start tuning (may take several minutes)
5. Review best parameters and scores
6. Use the optimized parameters for retraining

**Supported Algorithms**:
- **Random Forest**: n_estimators, max_depth, min_samples_split
- **SVM**: C, kernel, gamma

**Parameter Grids**:

```python
# Random Forest
{
    "n_estimators": [50, 100, 200],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10]
}

# SVM
{
    "C": [0.1, 1, 10],
    "kernel": ["rbf", "linear"],
    "gamma": ["scale", "auto"]
}
```

### 4. Model Retraining

**Purpose**: Train new models with custom configurations and different datasets.

**Capabilities**:
- Train on different NASA mission datasets
- Use default or custom hyperparameters
- Automatic model registration and loading
- Performance metrics display
- Integration with hyperparameter tuning results

**API Endpoint**: `POST /model/retrain`

**How to Use**:
1. Navigate to Advanced Features → Model Retraining
2. Select algorithm (Random Forest, Neural Network, or SVM)
3. Choose training dataset (Kepler, TESS, or K2)
4. Optionally provide custom hyperparameters (JSON format)
5. Start retraining (may take several minutes)
6. Review performance metrics
7. New model is automatically loaded for inference

**Supported Algorithms**:
- Random Forest
- Neural Network
- Support Vector Machine (SVM)

**Supported Datasets**:
- Kepler Mission (`cumulative.csv`)
- TESS Mission (`toi.csv`)
- K2 Mission (`k2targets.csv`)

**Example Custom Parameters**:

```json
{
  "n_estimators": 200,
  "max_depth": 30,
  "min_samples_split": 5,
  "random_state": 42
}
```

### 5. Educational Content

**Purpose**: Learn about exoplanets, detection methods, and NASA missions.

**Capabilities**:
- Comprehensive overview of exoplanets
- Explanation of detection methods (transit method)
- Planet type classifications (Earth-like, Super-Earth, Neptune-like, Jupiter-like)
- Feature explanations (orbital period, transit duration, etc.)
- NASA mission information (Kepler, TESS, K2)
- Future of exoplanet discovery

**API Endpoint**: `GET /education/exoplanet-info`

**How to Use**:
1. Navigate to Advanced Features → Learn About Exoplanets
2. Browse through different sections:
   - Overview: What are exoplanets?
   - Detection Methods: How we find them
   - Planet Types: Different categories
   - Features: Understanding the data
   - Missions: NASA's exoplanet programs

**Content Sections**:
- **Overview**: Introduction to exoplanets and their significance
- **Detection Methods**: Transit method explanation with key features
- **Planet Types**: Classification by size and characteristics
- **Features**: Detailed explanations of astronomical measurements
- **Missions**: History and achievements of Kepler, TESS, and K2

## Technical Implementation

### Backend (Python/FastAPI)

**New Endpoints**:
- `/model/feature-importance` - Get feature importance scores
- `/education/exoplanet-info` - Get educational content
- `/datasets/comparison` - Compare NASA mission datasets
- `/model/tune-hyperparameters` - Perform hyperparameter tuning
- `/model/retrain` - Retrain models with custom configuration

**Key Files**:
- `api/main.py` - API endpoint definitions
- `api/inference_service.py` - Feature importance extraction
- `models/model_trainer.py` - Training orchestration

### Frontend (Next.js/React)

**New Pages**:
- `/advanced` - Main advanced features page with tabbed interface

**New Components**:
- `FeatureImportanceView.tsx` - Feature importance visualization
- `DatasetComparisonView.tsx` - Dataset comparison charts
- `HyperparameterTuning.tsx` - Hyperparameter tuning interface
- `ModelRetraining.tsx` - Model retraining interface
- `ExoplanetEducation.tsx` - Educational content display

**Key Files**:
- `frontend/app/advanced/page.tsx` - Advanced features page
- `frontend/lib/api.ts` - API client functions
- `frontend/lib/types.ts` - TypeScript type definitions

## Usage Examples

### Example 1: Optimize and Retrain a Model

1. Go to Advanced Features → Hyperparameter Tuning
2. Select "Random Forest"
3. Click "Start Hyperparameter Tuning"
4. Wait for results (best parameters will be displayed)
5. Go to Model Retraining tab
6. Select "Random Forest" and "Kepler" dataset
7. Enable "Use Custom Hyperparameters"
8. Paste the best parameters from tuning
9. Click "Start Model Retraining"
10. New optimized model is now loaded and ready to use

### Example 2: Compare Mission Datasets

1. Ensure datasets are downloaded in `data/raw/`
2. Go to Advanced Features → Dataset Comparison
3. View statistics for each mission
4. Compare confirmation rates
5. Learn about mission characteristics
6. Use insights to choose best dataset for training

### Example 3: Understand Model Decisions

1. Go to Advanced Features → Feature Importance
2. View which features are most important
3. Note the top 5 features
4. Go to Learn About Exoplanets → Features
5. Read detailed explanations of important features
6. Use this knowledge when classifying new observations

## Performance Considerations

### Hyperparameter Tuning
- **Time**: 5-15 minutes depending on parameter grid size
- **Memory**: Requires loading full dataset into memory
- **CPU**: Uses all available cores (n_jobs=-1)

### Model Retraining
- **Time**: 
  - Random Forest: 1-3 minutes
  - Neural Network: 3-10 minutes
  - SVM: 2-5 minutes
- **Memory**: Depends on dataset size
- **Storage**: Each model is saved to registry (~1-50 MB)

### Dataset Comparison
- **Time**: < 1 second (cached after first load)
- **Memory**: Minimal (only loads dataset metadata)

## Troubleshooting

### Feature Importance Shows Uniform Values
- This is expected for Neural Networks (no built-in feature importance)
- Use permutation importance for more detailed analysis (future enhancement)

### Hyperparameter Tuning Fails
- Ensure dataset is available in `data/raw/`
- Check that sufficient memory is available
- Try reducing parameter grid size

### Model Retraining Fails
- Verify dataset file exists and is valid
- Check hyperparameter JSON syntax if using custom parameters
- Ensure sufficient disk space for model storage

### Dataset Comparison Shows No Data
- Download datasets using `python data/dataset_downloader.py`
- Verify files are in `data/raw/` directory
- Check file names match expected format

## Future Enhancements

Potential improvements for future versions:

1. **Permutation Importance**: More accurate feature importance for all algorithms
2. **Bayesian Optimization**: More efficient hyperparameter search
3. **Ensemble Models**: Combine multiple models for better performance
4. **Real-time Training**: Stream training progress to frontend
5. **Model Comparison**: Side-by-side comparison of multiple models
6. **Export Results**: Download tuning and training results as reports
7. **Advanced Visualizations**: SHAP values, partial dependence plots
8. **Custom Datasets**: Upload and train on user-provided data

## References

- NASA Exoplanet Archive: https://exoplanetarchive.ipac.caltech.edu/
- Kepler Mission: https://www.nasa.gov/mission_pages/kepler/main/index.html
- TESS Mission: https://www.nasa.gov/tess-transiting-exoplanet-survey-satellite
- K2 Mission: https://www.nasa.gov/mission_pages/kepler/main/index.html

## Support

For issues or questions about advanced features:
1. Check this documentation
2. Review the troubleshooting section
3. Check API logs for error messages
4. Verify all dependencies are installed
