"""
Verify downloaded NASA datasets are properly formatted and loadable.
"""

import pandas as pd
from pathlib import Path

def find_header_row(filepath):
    """Find the actual header row by skipping comment lines."""
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            # NASA files often have comments starting with #
            if not line.startswith('#'):
                return i
    return 0

def verify_dataset(filepath, dataset_name):
    """Verify a single dataset can be loaded."""
    print(f"\n{'='*60}")
    print(f"Verifying {dataset_name} Dataset")
    print(f"{'='*60}")
    
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return False
    
    print(f"✅ File exists: {filepath}")
    print(f"   Size: {filepath.stat().st_size / 1024 / 1024:.2f} MB")
    
    try:
        # Find where the actual data starts
        skip_rows = find_header_row(filepath)
        if skip_rows > 0:
            print(f"   Skipping {skip_rows} comment lines")
        
        # Load the dataset
        df = pd.read_csv(filepath, skiprows=skip_rows, low_memory=False)
        
        print(f"✅ Successfully loaded!")
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {len(df.columns)}")
        print(f"   First 5 columns: {list(df.columns[:5])}")
        
        # Check for disposition column
        disposition_cols = ['koi_disposition', 'tfopwg_disp', 'k2_disposition', 'pl_rade']
        found_disp = [col for col in disposition_cols if col in df.columns]
        
        if found_disp:
            print(f"   Disposition column: {found_disp[0]}")
            if found_disp[0] in df.columns:
                print(f"   Disposition values: {df[found_disp[0]].value_counts().to_dict()}")
        else:
            print(f"   ⚠️  No standard disposition column found")
            print(f"   Available columns: {list(df.columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

def main():
    """Verify all NASA datasets."""
    print("\n" + "="*60)
    print("NASA DATASET VERIFICATION")
    print("="*60)
    
    data_dir = Path('data/raw')
    
    datasets = [
        (data_dir / 'cumulative.csv', 'Kepler (KOI)'),
        (data_dir / 'toi.csv', 'TESS (TOI)'),
        (data_dir / 'k2targets.csv', 'K2'),
    ]
    
    results = {}
    for filepath, name in datasets:
        results[name] = verify_dataset(filepath, name)
    
    # Summary
    print(f"\n{'='*60}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*60}")
    
    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} datasets verified successfully")
    
    if passed == total:
        print("\n🎉 All datasets are ready for training!")
    else:
        print("\n⚠️  Some datasets need attention")

if __name__ == '__main__':
    main()
