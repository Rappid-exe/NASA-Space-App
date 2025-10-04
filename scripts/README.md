# Training Scripts

## 🎯 Which Script to Use?

### ⭐ **RECOMMENDED: train_no_normalization.py**

This is the **current production model** that works with the web interface.

```bash
python scripts/train_no_normalization.py
```

**Features:**
- Trains on real NASA Kepler data
- NO normalization (works directly with raw features)
- 87.85% F1 score, 82.43% accuracy
- Compatible with frontend (5 input features + 6 derived)
- Registers as `raw_randomforest`, `raw_neuralnetwork`, `raw_svm`

---

## 📋 Other Scripts (For Reference)

### train_with_real_nasa_data.py
- Uses 10 input features (includes stellar properties)
- Higher accuracy but incompatible with frontend
- Requires features not collected by UI

### train_simple_model.py
- Uses normalization during training
- Requires applying same normalization during inference
- More complex deployment

### quick_train.py
- Uses synthetic data for testing
- Fast training for development
- Not for production use

---

## 🔄 Retraining the Model

If you need to retrain with new data:

1. **Download NASA Data**
   ```bash
   # Place cumulative KOI CSV in root directory
   # Download from: https://exoplanetarchive.ipac.caltech.edu/
   ```

2. **Run Training**
   ```bash
   python scripts/train_no_normalization.py
   ```

3. **Restart API**
   ```bash
   python -m uvicorn api.main:app --reload
   ```

The API will automatically load the new `raw_randomforest` model.

---

## 📊 Expected Output

```
Training: 3,565 samples
Validation: 510 samples
Test: 1,019 samples

RandomForest:
  Accuracy:  0.8243
  Precision: 0.8480
  Recall:    0.9113
  F1 Score:  0.8785

Model registered: raw_randomforest_v1_YYYYMMDD_HHMMSS
```

---

## 🎓 Training Details

**Dataset:** NASA Kepler Objects of Interest (KOI)
**Source:** https://exoplanetarchive.ipac.caltech.edu/
**Observations:** ~5,000 after cleaning
**Features:** 11 total (5 input + 6 derived)
**Algorithms:** Random Forest, Neural Network, SVM
**Best Model:** Random Forest (87.85% F1)

---

## 🐛 Troubleshooting

### "File not found: cumulative_*.csv"
Download the NASA Kepler KOI dataset and place it in the root directory.

### "No module named 'models'"
Run from the project root directory, not from scripts/.

### "Model already exists"
The script will create a new version automatically.

---

**For more details, see the main [README.md](../README.md)**
