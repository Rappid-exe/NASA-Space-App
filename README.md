# 🪐 Exoplanet Classifier

[![NASA Space Apps Challenge 2025](https://img.shields.io/badge/NASA-Space%20Apps%20Challenge%202025-blue)](https://www.spaceappschallenge.org/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Next.js 14](https://img.shields.io/badge/Next.js-14-black)](https://nextjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![F1 Score](https://img.shields.io/badge/F1%20Score-87.85%25-success)](docs/TRAINING_RESULTS_REAL_NASA_DATA.md)

AI-powered system for identifying and classifying exoplanets using machine learning trained on NASA's Kepler mission data.

## ✨ Features

- 🎯 **Real-time Classification** - Classify exoplanet candidates instantly
- 📊 **Batch Processing** - Process thousands of observations via CSV upload
- 🧠 **Multiple ML Models** - Random Forest, Neural Networks, and SVM
- 📈 **Interactive Dashboard** - Visualize model performance metrics
- 🌐 **Modern Web Interface** - Beautiful Next.js frontend
- 🚀 **REST API** - FastAPI backend for easy integration
- ⚙️ **Advanced Features** - Hyperparameter tuning, model retraining, feature importance
- 📚 **Educational Content** - Learn about exoplanets and detection methods
- 🔬 **Dataset Comparison** - Compare discoveries across NASA missions

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 18+
- pip and npm

**Note**: The NASA Kepler dataset (~10MB) is not included in this repository. See [Data Download Guide](docs/DATA_DOWNLOAD.md) for instructions.

### 1. Install Backend Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download NASA Data & Train Model

**Download Kepler Data:**
1. Visit [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
2. Download the Kepler Objects of Interest (KOI) cumulative table as CSV
3. Place it in the project root directory

**Train the Model:**
```bash
python scripts/train_no_normalization.py
```

This trains on real NASA Kepler data and registers the model for use by the API.

### 3. Start the API Server

```bash
python -m uvicorn api.main:app --reload
```

API will be available at http://localhost:8000

### 4. Start the Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

Frontend will be available at http://localhost:3000

## 📁 Project Structure

```
NASA-Space-App/
├── api/              # FastAPI backend
│   ├── main.py      # API endpoints
│   └── inference_service.py
├── data/            # Data processing pipeline
│   ├── dataset_downloader.py
│   ├── dataset_loader.py
│   └── data_processor.py
├── models/          # ML model implementations
│   ├── base_classifier.py
│   ├── random_forest_classifier.py
│   ├── neural_network_classifier.py
│   ├── model_trainer.py
│   └── model_registry.py
├── frontend/        # Next.js web interface
│   ├── app/        # Pages
│   ├── components/ # React components
│   └── lib/        # API client
├── tests/          # Test files
├── examples/       # Example scripts
└── docs/           # Documentation
```

## 📚 Documentation

- **[Setup Guide](SETUP.md)** - Detailed installation instructions
- **[API Documentation](docs/API_QUICKSTART.md)** - API usage guide
- **[Frontend Guide](docs/FRONTEND_SETUP_GUIDE.md)** - Frontend setup
- **[Advanced Features](docs/ADVANCED_FEATURES.md)** - Hyperparameter tuning, retraining, and more
- **[Full Documentation](docs/)** - Complete documentation index

## 🎯 Usage

### Classify a Single Observation

```python
import requests

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

### Use the Web Interface

1. Open http://localhost:3000
2. Navigate to "Start Classifying"
3. Enter observation data or load an example
4. Click "Classify" to see results

### Upload CSV for Batch Processing

1. Go to http://localhost:3000/upload
2. Download the sample CSV template
3. Upload your CSV file
4. View and export results

## 🧪 Testing

```bash
# Test data ingestion
python tests/test_data_ingestion.py

# Test data processing
python tests/test_data_processing.py

# Test model training
python tests/test_model_training.py

# Test model registry
python tests/test_model_registry.py

# Test API
python tests/test_api_structure.py
```

## 🛠️ Tech Stack

### Backend
- **Python 3.8+** - Core language
- **FastAPI** - REST API framework
- **scikit-learn** - ML algorithms
- **TensorFlow** - Neural networks
- **pandas** - Data processing

### Frontend
- **Next.js 14** - React framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **Recharts** - Data visualization
- **Axios** - HTTP client

## 📊 Model Performance

**Trained on Real NASA Kepler Data** 🛰️

Current production model (Random Forest):
- **F1 Score**: 87.85%
- **Accuracy**: 82.43%
- **Precision**: 84.80%
- **Recall**: 91.13%

Trained on **5,094 real observations** from NASA's Kepler mission. Performance is comparable to published research (Ansdell et al. 2018: 85-90% with Random Forest on Kepler data).

## 🌌 Data Sources

**Currently Using**: Real NASA Kepler KOI data (9,564 observations)

- **Kepler Mission** - KOI (Kepler Objects of Interest) ✅ **Active**
- **TESS Mission** - TOI (TESS Objects of Interest)
- **K2 Mission** - K2 Candidates

Data from [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)

**Dataset File**: `cumulative_2025.10.04_10.12.10.csv` (included in repo)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- NASA Exoplanet Archive for datasets
- Kepler, TESS, and K2 mission teams
- Open source community

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Built with ❤️ for space exploration and AI** 🚀🪐
