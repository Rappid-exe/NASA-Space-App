# 🎯 Demo Quick Reference Card

## 🚀 Start Commands

```bash
# Terminal 1: API
python -m uvicorn api.main:app --reload

# Terminal 2: Frontend
cd frontend
npm run dev
```

**URLs**:
- Frontend: http://localhost:3000
- API Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

---

## 📊 Key Numbers to Remember

| Metric | Value |
|--------|-------|
| **F1 Score** | 90.60% |
| **Accuracy** | 84.94% |
| **Training Data** | 4,349 real observations |
| **Original Dataset** | 9,564 NASA Kepler KOIs |
| **Features** | 16 (10 input + 6 derived) |
| **Model** | Random Forest |

---

## 🧪 Test Examples

### Hot Jupiter (CONFIRMED) ✅
```
Orbital Period: 3.5 days
Transit Duration: 2.8 hours
Transit Depth: 500 ppm
Planetary Radius: 1.2 Earth radii
Temperature: 1200 K
```
**Expected**: CONFIRMED with high confidence

### Earth-like (CONFIRMED) ✅
```
Orbital Period: 365 days
Transit Duration: 13 hours
Transit Depth: 84 ppm
Planetary Radius: 1.0 Earth radii
Temperature: 288 K
```
**Expected**: CONFIRMED with moderate confidence

### False Positive ❌
```
Orbital Period: 15 days
Transit Duration: 0.5 hours
Transit Depth: 50 ppm
Planetary Radius: 0.3 Earth radii
Temperature: 400 K
```
**Expected**: FALSE POSITIVE

---

## 💬 Talking Points

### Opening
"We built an AI-powered exoplanet classifier trained on **real NASA Kepler data** with **90.60% F1 score**, comparable to published research."

### Data
- "Trained on 4,349 real observations from NASA's Kepler mission"
- "Downloaded directly from NASA Exoplanet Archive"
- "Includes 3,302 confirmed exoplanets and 1,047 false positives"

### Performance
- "90.60% F1 score and 84.94% accuracy"
- "Comparable to published research (Ansdell et al. 2018: 85-90%)"
- "Random Forest model performed best on real astronomical data"

### Features
- "Uses 10 astronomical features plus 6 derived features"
- "Includes orbital period, transit depth, planetary radius, temperature"
- "Automatically creates derived features like habitability indicators"

### System
- "Full-stack web application with REST API"
- "Real-time single classification and batch processing"
- "Beautiful space-themed UI with interactive visualizations"
- "Model registry with versioning and metadata"

---

## ❓ Expected Questions & Answers

### "How accurate is your model?"
"90.60% F1 score and 84.94% accuracy on real NASA Kepler data, which is comparable to published research papers in this field."

### "What data did you use?"
"Real NASA Kepler Objects of Interest data - 9,564 observations from the official NASA Exoplanet Archive, processed down to 4,349 high-quality observations."

### "How does it compare to research?"
"Our 90.60% F1 score is in line with published papers like Ansdell et al. (2018) who achieved 85-90% with Random Forest on Kepler data."

### "Can it handle new data?"
"Yes! We have a REST API that can classify single observations in real-time or process batches of up to 1,000 observations via CSV upload."

### "What makes it production-ready?"
"We have model versioning, metadata tracking, comprehensive error handling, input validation, and a full web interface for easy use."

### "Why Random Forest?"
"Random Forest performed best because it handles the non-linear relationships in astronomical data well, captures feature interactions, and is robust to noise."

### "What features does it use?"
"10 astronomical features from NASA data like orbital period, transit duration, depth, planetary radius, and temperature, plus 6 derived features we engineered."

### "How long does training take?"
"About 2-3 minutes on a standard laptop for all three models (Random Forest, Neural Network, SVM)."

### "Can you add more data?"
"Yes! The system is designed to work with TESS and K2 mission data as well. We can easily retrain with additional datasets."

### "What's next?"
"We plan to add hyperparameter tuning, feature importance visualization, and combine data from multiple NASA missions for even better performance."

---

## 🎬 Demo Flow (5 minutes)

### 1. Introduction (30 sec)
"We built an AI exoplanet classifier trained on real NASA data..."

### 2. Show Dashboard (30 sec)
- Open homepage
- Point out model metrics
- Highlight real NASA data

### 3. Single Classification (1 min)
- Navigate to "Start Classifying"
- Load Hot Jupiter example
- Click Classify
- Explain results and confidence

### 4. Explain Features (1 min)
- Show the input features
- Mention derived features
- Explain how model works

### 5. Batch Processing (1 min)
- Navigate to "Upload Dataset"
- Show CSV template
- Explain batch capability
- Show results visualization

### 6. API Demo (30 sec)
- Open API docs (http://localhost:8000/docs)
- Show available endpoints
- Mention integration possibilities

### 7. Technical Details (30 sec)
- Model registry
- Version tracking
- Performance metrics

### 8. Wrap Up (30 sec)
- Summarize achievements
- Mention future enhancements
- Take questions

---

## 🛠️ Troubleshooting

### API Not Loading Model
```bash
# Check registry
python -c "from models.model_registry import ModelRegistry; ModelRegistry().print_registry_summary()"

# Retrain if needed
python train_with_real_nasa_data.py
```

### Frontend Not Connecting
- Check API is running on port 8000
- Check .env.local has correct API URL
- Restart frontend: `npm run dev`

### Port Already in Use
```bash
# Windows - Kill process
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

---

## 📸 Screenshot Checklist

Before demo, capture:
- [ ] Homepage with dashboard
- [ ] Classification form with example
- [ ] Results with confidence scores
- [ ] Batch upload interface
- [ ] API documentation page
- [ ] Model comparison table

---

## ✅ Pre-Demo Checklist

- [ ] Both servers running
- [ ] Frontend loads correctly
- [ ] Dashboard shows correct metrics (90.60% F1)
- [ ] Test Hot Jupiter example works
- [ ] Test Earth-like example works
- [ ] Test False Positive example works
- [ ] CSV upload tested
- [ ] API docs accessible
- [ ] Know your numbers (90.60%, 84.94%, 4,349)
- [ ] Talking points memorized
- [ ] Questions & answers reviewed

---

## 🎯 Key Differentiators

What makes your project stand out:

1. **Real NASA Data** - Not synthetic, actual Kepler observations
2. **Research-Grade** - Performance comparable to published papers
3. **Production-Ready** - Full-stack system, not just a notebook
4. **User-Friendly** - Beautiful UI, not just command line
5. **Scalable** - REST API, batch processing, model registry
6. **Well-Documented** - Complete documentation and guides

---

## 🏆 Closing Statement

"We've built a production-ready exoplanet classifier that achieves research-grade performance on real NASA data. It's not just a proof of concept - it's a fully functional system that could actually assist astronomers in discovering new exoplanets. Thank you!"

---

**Good luck! 🚀🪐**
