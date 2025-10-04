# 🚀 Start the Exoplanet Classifier

## ✅ Model Trained Successfully!

- **Algorithm**: NeuralNetwork
- **Accuracy**: 96.92%
- **F1 Score**: 94.75%
- **Model ID**: exoplanet_classifier_v1_20251004_175146

## 🎯 Start the Application

### Terminal 1: Start Backend API

```bash
python -m uvicorn api.main:app --reload
```

The API will start at: **http://localhost:8000**
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### Terminal 2: Start Frontend

```bash
cd frontend
npm run dev
```

The frontend will start at: **http://localhost:3000**

## 🧪 Test It Out

1. Open http://localhost:3000 in your browser
2. You should see the homepage with the dashboard
3. Click "Start Classifying" to classify observations
4. Try the example values (Hot Jupiter, Earth-like, etc.)
5. Go to "Upload Dataset" to test batch processing

## 📊 What You Can Do

- **Homepage**: View model statistics and performance metrics
- **Classify Page**: Enter observation data and get real-time predictions
- **Upload Page**: Upload CSV files for batch classification

## ✅ Verification Checklist

- [ ] Backend API running on port 8000
- [ ] Frontend running on port 3000
- [ ] Homepage loads without errors
- [ ] Dashboard shows model statistics (96.92% accuracy)
- [ ] Can classify single observations
- [ ] Can upload CSV files
- [ ] Results display correctly

## 🐛 Troubleshooting

**API won't start:**
- Check if port 8000 is already in use
- Make sure you're in the project root directory

**Frontend won't start:**
- Run `npm install` first if you haven't
- Check if port 3000 is already in use
- Make sure you're in the `frontend/` directory

**"No model loaded" error:**
- The model should be automatically loaded
- Check `models/registry/` directory exists
- Re-run `python quick_train.py` if needed

## 🎉 You're Ready!

Your exoplanet classifier is now running end-to-end:
- ✅ Model trained (96.92% accuracy)
- ✅ API ready for inference
- ✅ Frontend ready for user interaction

**Start both servers and open http://localhost:3000!** 🪐
