# Changelog

All notable changes to the Exoplanet Classifier project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-10-04

### 🎉 Initial Release - NASA Space Apps Challenge 2025

#### Added
- **Real NASA Data Integration**
  - Support for Kepler Objects of Interest (KOI) dataset
  - 5,094 real observations after cleaning
  - Automatic data processing pipeline

- **Machine Learning Models**
  - Random Forest classifier (87.85% F1 score) ⭐ Production model
  - Neural Network classifier (87.70% F1 score)
  - Support Vector Machine (87.44% F1 score)
  - Model registry with versioning
  - Model persistence and loading

- **REST API (FastAPI)**
  - Single observation classification endpoint
  - Batch classification endpoint (up to 1,000 observations)
  - Model statistics endpoint
  - Health check endpoint
  - Automatic model loading
  - Input validation

- **Web Interface (Next.js 14)**
  - Interactive dashboard with model metrics
  - Single observation classification form
  - CSV batch upload with drag-and-drop
  - Results visualization with charts
  - Example data presets (Hot Jupiter, Earth-like, False Positive)
  - Responsive design for mobile/tablet/desktop
  - Space-themed UI with animations

- **Data Processing**
  - Missing value handling
  - Outlier removal (IQR method)
  - Feature engineering (6 derived features)
  - Feature normalization options
  - Support for multiple NASA datasets

- **Documentation**
  - Comprehensive README
  - Setup guide (SETUP.md)
  - API documentation
  - Frontend documentation
  - Data download guide
  - Training scripts documentation
  - Contributing guidelines

#### Technical Details
- **Backend**: Python 3.8+, FastAPI, scikit-learn, TensorFlow
- **Frontend**: Next.js 14, TypeScript, Tailwind CSS, Recharts
- **Data**: NASA Kepler KOI dataset (9,564 observations)
- **Performance**: 87.85% F1 score, 82.43% accuracy
- **Features**: 11 total (5 input + 6 derived)

#### Known Limitations
- Binary classification only (CONFIRMED vs FALSE POSITIVE)
- Limited to 5 input features for web interface
- No real-time model retraining
- No hyperparameter tuning UI
- Single dataset support (Kepler only)

---

## [Unreleased]

### Planned Features
- [ ] Support for TESS and K2 datasets
- [ ] Feature importance visualization
- [ ] Hyperparameter tuning interface
- [ ] Model comparison dashboard
- [ ] Real-time model retraining
- [ ] Docker containerization
- [ ] Cloud deployment guides
- [ ] Comprehensive test suite
- [ ] CI/CD pipeline

---

## Version History

### [1.0.0] - 2025-10-04
- Initial release for NASA Space Apps Challenge 2025
- Full-stack exoplanet classification system
- Trained on real NASA Kepler data
- Production-ready web interface

---

**For detailed changes, see the [commit history](https://github.com/Rappid-exe/NASA-Space-App/commits/main)**
