# 🚀 Start Your Exoplanet Classifier (Real NASA Data)

## ✅ You're Ready!

Your exoplanet classifier has been trained on **real NASA Kepler data** and is ready to use!

### 🏆 Your Model Performance
- **Algorithm**: Random Forest
- **F1 Score**: 90.60%
- **Accuracy**: 84.94%
- **Training Data**: 4,349 real Kepler observations
- **Model ID**: `kepler_randomforest_v1_20251004_185318`

---

## 🎯 Quick Start (2 Steps)

### Step 1: Start the Backend API

Open a terminal and run:

```bash
python -m uvicorn api.main:app --reload
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
Loaded best model: kepler_randomforest_v1_20251004_185318
```

✅ API is running at: **http://localhost:8000**
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### Step 2: Start the Frontend

Open a **new terminal** and run:

```bash
cd frontend
npm run dev
```

You should see:
```
  ▲ Next.js 14.2.3
  - Local:        http://localhost:3000
```

✅ Frontend is running at: **http://localhost:3000**

---

## 🎮 Try It Out!

### 1. Open Your Browser
Navigate to: **http://localhost:3000**

### 2. View Dashboard
The homepage shows your model's performance metrics:
- Accuracy: 84.94%
- F1 Score: 90.60%
- Training on real NASA Kepler data

### 3. Classify Single Observations

Click **"Start Classifying"** and try these examples:

#### Example 1: Hot Jupiter (Should be CONFIRMED)
```
Orbital Period: 3.5 days
Transit Duration: 2.8 hours
Transit Depth: 500 ppm
Planetary Radius: 1.2 Earth radii
Equilibrium Temperature: 1200 K
```

#### Example 2: Earth-like Planet (Should be CONFIRMED)
```
Orbital Period: 365 days
Transit Duration: 13 hours
Transit Depth: 84 ppm
Planetary Radius: 1.0 Earth radii
Equilibrium Temperature: 288 K
```

#### Example 3: False Positive
```
Orbital Period: 15.0 days
Transit Duration: 0.5 hours
Transit Depth: 50 ppm
Planetary Radius: 0.3 Earth radii
Equilibrium Temperature: 400 K
```

### 4. Batch Processing

Click **"Upload Dataset"** to:
1. Download the sample CSV template
2. Add your observations
3. Upload and get batch results
4. Export results as CSV

---

## 🧪 Test the API Directly

### Using cURL

```bash
# Test health endpoint
curl http://localhost:8000/health

# Classify an observation
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "orbital_period": 3.5,
    "transit_duration": 2.8,
    "transit_depth": 500.0,
    "planetary_radius": 1.2,
    "equilibrium_temperature": 1200.0
  }'
```

### Using Python

```python
import requests

response = requests.post(
    "http://localhost:8000/classify",
    json={
        "orbital_period": 3.5,
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

### Interactive API Docs

Visit: **http://localhost:8000/docs**

Try out all endpoints interactively with Swagger UI!

---

## 📊 What Your Model Can Do

### ✅ Real-Time Classification
- Classify individual exoplanet candidates
- Get confidence scores and probabilities
- Receive human-readable explanations

### ✅ Batch Processing
- Upload CSV files with multiple observations
- Process up to 1,000 observations at once
- Export results for further analysis

### ✅ Model Information
- View current model performance metrics
- See training dataset information
- Check model version and algorithm

---

## 🎯 Understanding Results

### Prediction Types

**CONFIRMED** - High confidence exoplanet
- Typical characteristics:
  - Regular orbital period
  - Consistent transit depth
  - Appropriate planetary radius
  - Reasonable temperature

**FALSE_POSITIVE** - Not an exoplanet
- Could be:
  - Binary star system
  - Instrumental noise
  - Background eclipsing binary
  - Stellar activity

### Confidence Levels

- **Very High**: > 95% - Strong signal
- **High**: 85-95% - Good confidence
- **Moderate**: 70-85% - Uncertain
- **Low**: < 70% - Needs review

---

## 🔧 Troubleshooting

### API Won't Start

**Error**: `Address already in use`
```bash
# Kill process on port 8000
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Then restart:
python -m uvicorn api.main:app --reload
```

**Error**: `No module named 'fastapi'`
```bash
pip install -r requirements.txt
```

### Frontend Won't Start

**Error**: `Port 3000 already in use`
```bash
# Use a different port:
cd frontend
npm run dev -- -p 3001
```

**Error**: `Cannot find module`
```bash
cd frontend
npm install
npm run dev
```

### Model Not Loading

**Error**: `No models found in registry`
```bash
# Retrain the model:
python train_with_real_nasa_data.py
```

---

## 📈 Model Registry

### View All Models

```python
from models.model_registry import ModelRegistry

registry = ModelRegistry("models/registry")
registry.print_registry_summary()
```

### Compare Models

```python
registry.compare_models()
```

### Load Specific Model

```python
# Load Random Forest (best)
model, metadata = registry.load_model(model_name="kepler_randomforest")

# Load Neural Network
model, metadata = registry.load_model(model_name="kepler_neuralnetwork")

# Load SVM
model, metadata = registry.load_model(model_name="kepler_svm")
```

---

## 🎓 For Your Hackathon Presentation

### Key Points to Highlight

1. **Real NASA Data** 🛰️
   - Trained on 4,349 real Kepler observations
   - Official NASA Exoplanet Archive data
   - Includes confirmed exoplanets and false positives

2. **Strong Performance** 📊
   - 90.60% F1 Score
   - 84.94% Accuracy
   - Comparable to published research

3. **Production-Ready** 🚀
   - Full-stack web application
   - REST API for integration
   - Batch processing capability
   - Model versioning and registry

4. **User-Friendly** 🎨
   - Beautiful space-themed UI
   - Interactive visualizations
   - Example data provided
   - CSV upload for batch processing

### Demo Flow

1. **Show Dashboard** - Model metrics and performance
2. **Classify Examples** - Hot Jupiter, Earth-like, False Positive
3. **Explain Results** - Confidence scores and reasoning
4. **Batch Upload** - Process multiple observations
5. **Show API Docs** - Integration capabilities

---

## 📚 Documentation

- **[README.md](README.md)** - Project overview
- **[SETUP.md](SETUP.md)** - Detailed setup instructions
- **[TRAINING_RESULTS_REAL_NASA_DATA.md](TRAINING_RESULTS_REAL_NASA_DATA.md)** - Training results
- **[REAL_NASA_DATA_TRAINING.md](REAL_NASA_DATA_TRAINING.md)** - Data processing details
- **[docs/](docs/)** - Complete documentation

---

## 🎉 You're All Set!

Your exoplanet classifier is:
- ✅ Trained on real NASA data
- ✅ Production-ready
- ✅ Fully functional
- ✅ Ready to demo

**Start both servers and open http://localhost:3000!** 🪐🚀

---

## 💡 Tips

### For Best Results
- Use realistic values for exoplanet parameters
- Check confidence scores - high confidence = reliable prediction
- Try the example values provided in the UI

### For Hackathon
- Emphasize the use of real NASA data
- Show the comparison with published research
- Demonstrate both single and batch classification
- Highlight the model registry and versioning

### For Future Development
- Add more NASA datasets (TESS, K2)
- Implement hyperparameter tuning
- Add feature importance visualization
- Create model comparison interface

---

**Good luck with your hackathon! 🏆**
