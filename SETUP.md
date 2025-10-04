# Setup Guide

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Installation

Test that all required libraries are installed:

```bash
python -c "import pandas, sklearn, tensorflow, requests; print('All dependencies installed successfully!')"
```

### 3. Run Data Ingestion Test

Download and inspect NASA datasets:

```bash
python test_data_ingestion.py
```

## What Gets Created

When you run the data ingestion test, the following happens:

1. **data/raw/** directory is created automatically
2. Three datasets are downloaded:
   - `koi_dataset.csv` - Kepler Objects of Interest
   - `toi_dataset.csv` - TESS Objects of Interest  
   - `k2_dataset.csv` - K2 mission candidates

3. Each dataset is:
   - Loaded into memory
   - Inspected for structure and content
   - Validated for integrity
   - Analyzed for classification distributions

## Troubleshooting

### Import Errors

If you get import errors, ensure you're using Python 3.8 or higher:

```bash
python --version
```

### Download Failures

If dataset downloads fail:
- Check your internet connection
- NASA Exoplanet Archive may be temporarily unavailable
- Try running the script again after a few minutes

### Memory Issues

The datasets can be large. If you encounter memory issues:
- Close other applications
- The system uses `low_memory=False` for proper type inference
- Consider using a machine with at least 4GB RAM

## Directory Structure After Setup

```
.
├── data/
│   ├── raw/
│   │   ├── koi_dataset.csv
│   │   ├── toi_dataset.csv
│   │   └── k2_dataset.csv
│   ├── dataset_downloader.py
│   ├── dataset_loader.py
│   └── dataset_validator.py
├── models/
├── api/
├── web/
├── requirements.txt
├── test_data_ingestion.py
├── README.md
└── SETUP.md
```

## Next Steps

After successful setup:
1. Review the dataset inspection output
2. Proceed to Task 2: Data processing and feature engineering
3. Begin building ML models in Task 3
