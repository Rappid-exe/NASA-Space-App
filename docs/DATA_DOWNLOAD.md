# 📥 Downloading NASA Exoplanet Data

## Why Download Separately?

The NASA Kepler dataset is **~10MB** and changes frequently as new discoveries are made. We don't include it in the repository to:
- Keep the repo size small
- Allow you to get the latest data
- Comply with GitHub file size limits

---

## 🛰️ Option 1: NASA Exoplanet Archive (Recommended)

### Step 1: Visit the Archive
Go to: https://exoplanetarchive.ipac.caltech.edu/

### Step 2: Navigate to KOI Table
1. Click on **"Data"** in the top menu
2. Select **"Kepler Objects of Interest"** (KOI)
3. Or direct link: https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative

### Step 3: Download as CSV
1. Click **"Download Table"** button
2. Select **"CSV Format"**
3. Choose **"All Columns"** or at minimum these columns:
   - `koi_disposition`
   - `koi_period`
   - `koi_duration`
   - `koi_depth`
   - `koi_prad`
   - `koi_teq`

### Step 4: Save to Project
Save the file as `cumulative_YYYY.MM.DD_HH.MM.SS.csv` in your project root directory.

---

## 🔗 Option 2: Direct Download (Faster)

Use this direct download link (may become outdated):

```bash
# Windows PowerShell
Invoke-WebRequest -Uri "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv" -OutFile "cumulative.csv"

# Linux/Mac
curl "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv" -o cumulative.csv
```

---

## 📊 What You Should See

After downloading, you should have:
- **File size**: ~10-15 MB
- **Rows**: ~9,500-10,000 observations
- **Columns**: 40-50 columns

### Verify the Data

```bash
# Check file size
ls -lh cumulative*.csv

# Check row count (should be ~9,500)
wc -l cumulative*.csv

# View first few lines
head cumulative*.csv
```

Or in Python:
```python
import pandas as pd
df = pd.read_csv('cumulative_2025.10.04_10.12.10.csv', comment='#')
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(df['koi_disposition'].value_counts())
```

Expected output:
```
Rows: 9564
Columns: 49
koi_disposition
FALSE POSITIVE    4839
CONFIRMED         2746
CANDIDATE         1979
```

---

## 🎯 After Downloading

Once you have the data:

1. **Train the model:**
   ```bash
   python scripts/train_no_normalization.py
   ```

2. **Start the system:**
   ```bash
   # Terminal 1: API
   python -m uvicorn api.main:app --reload
   
   # Terminal 2: Frontend
   cd frontend && npm run dev
   ```

3. **Open browser:**
   http://localhost:3000

---

## 🔄 Updating the Data

NASA updates the KOI table as new discoveries are made. To get the latest data:

1. Download the new CSV from NASA
2. Replace the old CSV file
3. Retrain the model: `python scripts/train_no_normalization.py`
4. Restart the API

---

## 📝 Data Attribution

This data is from NASA's Exoplanet Archive and is in the public domain.

**Citation:**
```
This research has made use of the NASA Exoplanet Archive, which is operated
by the California Institute of Technology, under contract with the National
Aeronautics and Space Administration under the Exoplanet Exploration Program.
```

**More info:**
- Kepler Mission: https://www.nasa.gov/mission_pages/kepler/
- Exoplanet Archive: https://exoplanetarchive.ipac.caltech.edu/
- Data Documentation: https://exoplanetarchive.ipac.caltech.edu/docs/

---

## ❓ Troubleshooting

### "File not found" error
Make sure the CSV file is in the project root directory (same level as README.md).

### "No data loaded" error
Check that the CSV file isn't corrupted. Try re-downloading.

### "Wrong format" error
Make sure you downloaded the **cumulative** KOI table, not a different dataset.

---

**Need help?** Open an issue on GitHub!
