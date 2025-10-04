# Advanced Features Quick Start Guide

## Accessing Advanced Features

1. Start the API server:
   ```bash
   python -m uvicorn api.main:app --reload
   ```

2. Start the frontend:
   ```bash
   cd frontend
   npm run dev
   ```

3. Open http://localhost:3000

4. Click the **"⚙️ Advanced Features"** button on the home page

## Quick Tour

### 1. Feature Importance (30 seconds)

**What it does**: Shows which features are most important for predictions

**Steps**:
1. Click "Feature Importance" tab
2. View the top 5 most important features
3. Scroll down to see the full chart
4. Read feature explanations at the bottom

**Use case**: Understanding what drives the model's decisions

---

### 2. Dataset Comparison (1 minute)

**What it does**: Compares discoveries across NASA missions

**Steps**:
1. Click "Dataset Comparison" tab
2. View summary statistics at the top
3. Compare mission cards (Kepler, TESS, K2)
4. Explore the charts
5. Read about each mission

**Use case**: Understanding different NASA datasets

**Note**: Requires datasets in `data/raw/` directory

---

### 3. Hyperparameter Tuning (5-15 minutes)

**What it does**: Finds optimal model parameters

**Steps**:
1. Click "Hyperparameter Tuning" tab
2. Select algorithm (Random Forest or SVM)
3. Adjust CV folds (default: 5)
4. Click "Start Hyperparameter Tuning"
5. Wait for results (5-15 minutes)
6. Copy the best parameters

**Use case**: Optimizing model performance

**Tip**: Use the best parameters in Model Retraining

---

### 4. Model Retraining (2-10 minutes)

**What it does**: Trains a new model with custom settings

**Steps**:
1. Click "Model Retraining" tab
2. Select algorithm (Random Forest, Neural Network, or SVM)
3. Choose dataset (Kepler, TESS, or K2)
4. Optionally enable custom hyperparameters
5. Paste optimized parameters from tuning (if available)
6. Click "Start Model Retraining"
7. Wait for training to complete
8. View performance metrics

**Use case**: Creating optimized models or testing different datasets

**Note**: New model is automatically loaded for inference

---

### 5. Learn About Exoplanets (5 minutes)

**What it does**: Educational content about exoplanets

**Steps**:
1. Click "Learn About Exoplanets" tab
2. Browse through sections:
   - **Overview**: What are exoplanets?
   - **Detection**: How we find them
   - **Types**: Different planet categories
   - **Features**: Understanding the data
   - **Missions**: NASA's programs

**Use case**: Learning about exoplanet science

---

## Common Workflows

### Workflow 1: Optimize a Model

**Goal**: Create the best possible model

**Steps**:
1. Go to Hyperparameter Tuning
2. Select Random Forest
3. Run tuning (wait 5-15 min)
4. Copy best parameters
5. Go to Model Retraining
6. Select Random Forest + Kepler dataset
7. Enable custom hyperparameters
8. Paste best parameters
9. Start retraining
10. Test the new model on the Classify page

**Time**: 15-30 minutes

---

### Workflow 2: Compare Datasets

**Goal**: Understand which dataset is best

**Steps**:
1. Go to Dataset Comparison
2. Note confirmation rates for each mission
3. Go to Model Retraining
4. Train a model on Kepler dataset
5. Note the performance metrics
6. Train another model on TESS dataset
7. Compare the results
8. Choose the best performing dataset

**Time**: 10-20 minutes

---

### Workflow 3: Understand Model Decisions

**Goal**: Learn what features matter most

**Steps**:
1. Go to Feature Importance
2. Note the top 5 features
3. Go to Learn About Exoplanets → Features
4. Read about each important feature
5. Go to Classify page
6. Enter values for the important features
7. See how they affect predictions

**Time**: 5-10 minutes

---

## Tips and Tricks

### Hyperparameter Tuning
- Start with Random Forest (faster than SVM)
- Use 5 CV folds for balance between speed and accuracy
- Save the best parameters for later use

### Model Retraining
- Kepler dataset is largest and most reliable
- Neural Networks take longest to train
- Random Forest is fastest and often best performing

### Feature Importance
- Different algorithms may rank features differently
- Focus on the top 5 features for best results
- Use this to understand which data to collect

### Dataset Comparison
- Kepler has most confirmed exoplanets
- TESS focuses on nearby bright stars
- K2 has diverse fields of view

### Educational Content
- Start with Overview for context
- Read Detection Methods to understand the science
- Use Features section as a reference

---

## Troubleshooting

### "No model loaded" error
**Solution**: Train a model first using `python scripts/train_no_normalization.py`

### Hyperparameter tuning takes too long
**Solution**: This is normal. It can take 5-15 minutes. Be patient!

### Model retraining fails
**Solution**: 
- Check that dataset exists in `data/raw/`
- Verify JSON syntax if using custom parameters
- Ensure sufficient disk space

### Dataset comparison shows no data
**Solution**: Download datasets using `python data/dataset_downloader.py`

### Feature importance shows uniform values
**Solution**: This is expected for Neural Networks. Try Random Forest or SVM.

---

## Next Steps

After exploring advanced features:

1. **Test Your Model**: Go to the Classify page and test predictions
2. **Batch Processing**: Upload a CSV on the Upload page
3. **View Dashboard**: Check model statistics on the home page
4. **Experiment**: Try different algorithms and datasets
5. **Learn More**: Read the full documentation in `docs/ADVANCED_FEATURES.md`

---

## API Access

All features are also available via REST API:

```bash
# Feature Importance
curl http://localhost:8000/model/feature-importance

# Educational Content
curl http://localhost:8000/education/exoplanet-info

# Dataset Comparison
curl http://localhost:8000/datasets/comparison

# Hyperparameter Tuning
curl -X POST http://localhost:8000/model/tune-hyperparameters \
  -H "Content-Type: application/json" \
  -d '{"algorithm": "RandomForest", "param_grid": {...}, "cv_folds": 5}'

# Model Retraining
curl -X POST http://localhost:8000/model/retrain \
  -H "Content-Type: application/json" \
  -d '{"algorithm": "RandomForest", "dataset": "kepler"}'
```

See `docs/ADVANCED_FEATURES.md` for detailed API documentation.

---

**Happy exploring! 🚀🪐**
