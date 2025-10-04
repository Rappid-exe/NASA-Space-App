# Integration Quick Start Guide

## Overview

This guide provides quick commands to verify the complete integration of the exoplanet classification system.

## Prerequisites

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure Dataset is Available:**
   ```bash
   # Dataset should be at: data/raw/cumulative.csv
   # If not, copy from root: cumulative_2025.10.04_10.12.10.csv
   ```

3. **Verify Model is Trained:**
   ```bash
   # Check if models exist in: models/registry/
   # If not, run: python scripts/train_with_real_nasa_data.py
   ```

## Quick Integration Test

Run all integration tests to verify the complete system:

```bash
python -m pytest tests/test_integration_e2e.py -v
```

Expected output: **13 tests passed** ✅

## Step-by-Step Verification

### 1. Test Data Pipeline

```bash
python -m pytest tests/test_integration_e2e.py::TestEndToEndIntegration::test_01_data_loading_pipeline -v -s
```

**Expected:** ✅ Loaded 9,564 observations

### 2. Test Data Processing

```bash
python -m pytest tests/test_integration_e2e.py::TestEndToEndIntegration::test_02_data_processing_pipeline -v -s
```

**Expected:** ✅ Processed samples with 54 features

### 3. Test Model Training

```bash
python -m pytest tests/test_integration_e2e.py::TestEndToEndIntegration::test_03_model_training_pipeline -v -s
```

**Expected:** ✅ Model trained with accuracy > 0.5

### 4. Test Inference Service

```bash
python -m pytest tests/test_integration_e2e.py::TestEndToEndIntegration::test_05_inference_service_integration -v -s
```

**Expected:** ✅ Prediction: CONFIRMED or FALSE_POSITIVE

### 5. Test API Integration (requires API running)

**Start API Server:**
```bash
python api/main.py
```

**In another terminal, run:**
```bash
python -m pytest tests/test_integration_e2e.py::TestEndToEndIntegration::test_09_api_health_check -v -s
python -m pytest tests/test_integration_e2e.py::TestEndToEndIntegration::test_10_api_classification_endpoint -v -s
```

**Expected:** ✅ API health check passed, classification successful

### 6. Test Complete Pipeline

```bash
python -m pytest tests/test_integration_e2e.py::TestEndToEndIntegration::test_11_complete_pipeline_validation -v -s
```

**Expected:** ✅ Complete pipeline validation with all steps passing

### 7. Test Performance Benchmarks

```bash
python -m pytest tests/test_integration_e2e.py::TestModelPerformanceBenchmarks -v -s
```

**Expected:** 
- ✅ Accuracy > 70%
- ✅ F1 Score > 65%
- ✅ Consistent predictions

## Manual Integration Testing

### Test Backend API

1. **Start API Server:**
   ```bash
   python api/main.py
   ```

2. **Test Health Endpoint:**
   ```bash
   curl http://localhost:8000/health
   ```

3. **Test Classification:**
   ```bash
   curl -X POST http://localhost:8000/classify \
     -H "Content-Type: application/json" \
     -d '{
       "orbital_period": 3.52,
       "transit_duration": 2.8,
       "transit_depth": 500.0,
       "planetary_radius": 1.2,
       "equilibrium_temperature": 1200.0
     }'
   ```

4. **Test Model Statistics:**
   ```bash
   curl http://localhost:8000/model/statistics
   ```

### Test Frontend Integration

1. **Start Frontend (in frontend directory):**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

2. **Open Browser:**
   ```
   http://localhost:3000
   ```

3. **Test Features:**
   - ✅ Dashboard displays model statistics
   - ✅ Classification form accepts input
   - ✅ File upload works
   - ✅ Results display with confidence scores
   - ✅ Advanced features accessible

## Integration Checklist

Use this checklist to verify complete integration:

### Data Layer
- [ ] Dataset loads successfully
- [ ] Data processing pipeline works
- [ ] Feature engineering creates derived features
- [ ] Data splitting maintains stratification

### Model Layer
- [ ] Models train successfully
- [ ] Model evaluation produces metrics
- [ ] Model registry stores models
- [ ] Models can be loaded by ID or name

### Inference Layer
- [ ] Inference service loads models
- [ ] Single predictions work
- [ ] Batch predictions work
- [ ] Feature importance extraction works

### API Layer
- [ ] API server starts without errors
- [ ] Health endpoint responds
- [ ] Classification endpoint works
- [ ] Batch classification endpoint works
- [ ] Model statistics endpoint works
- [ ] Feature importance endpoint works
- [ ] Educational content endpoint works
- [ ] Dataset comparison endpoint works

### Frontend Layer
- [ ] Frontend builds successfully
- [ ] Dashboard displays correctly
- [ ] Classification form submits
- [ ] File upload processes
- [ ] Results display properly
- [ ] Advanced features work

### End-to-End
- [ ] Complete data flow works
- [ ] Frontend → API → Inference → Model
- [ ] Predictions match expectations
- [ ] Performance meets benchmarks

## Troubleshooting

### Issue: Tests fail with "No dataset available"

**Solution:**
```bash
# Copy dataset to correct location
copy cumulative_2025.10.04_10.12.10.csv data\raw\cumulative.csv
```

### Issue: Tests fail with "No model loaded"

**Solution:**
```bash
# Train a model first
python scripts/train_with_real_nasa_data.py
```

### Issue: API tests fail with "Connection refused"

**Solution:**
```bash
# Start API server in separate terminal
python api/main.py
```

### Issue: Frontend tests fail

**Solution:**
```bash
# Ensure API is running
python api/main.py

# In another terminal, start frontend
cd frontend
npm run dev
```

## Performance Expectations

### Data Processing
- **Time:** < 10 seconds for 9,564 observations
- **Memory:** < 500 MB

### Model Training
- **Time:** < 30 seconds for RandomForest (10 trees)
- **Memory:** < 1 GB

### Inference
- **Single Prediction:** < 100 ms
- **Batch (100 observations):** < 1 second

### API Response Times
- **Health Check:** < 50 ms
- **Classification:** < 200 ms
- **Batch Classification:** < 2 seconds (100 observations)

## Success Criteria

The integration is successful when:

✅ All 13 integration tests pass  
✅ Model accuracy > 70%  
✅ F1 Score > 65%  
✅ API responds to all endpoints  
✅ Frontend displays predictions  
✅ Complete pipeline processes data end-to-end  

## Next Steps

After verifying integration:

1. **Deploy to Production:**
   - Set up production environment
   - Configure environment variables
   - Deploy API and frontend

2. **Monitor Performance:**
   - Track API response times
   - Monitor model accuracy
   - Log prediction requests

3. **Continuous Integration:**
   - Run integration tests on every commit
   - Automate deployment pipeline
   - Set up performance monitoring

## Support

For issues or questions:
- Check `docs/INTEGRATION_TEST_SUMMARY.md` for detailed test results
- Review `tests/test_integration_e2e.py` for test implementation
- See `README.md` for general setup instructions
