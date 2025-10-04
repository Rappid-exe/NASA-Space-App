"""
Dataset downloader for NASA exoplanet datasets.
Handles downloading KOI, TOI, and K2 datasets from NASA archives.
"""

import os
import requests
from typing import Optional


class DatasetDownloader:
    """Downloads NASA exoplanet datasets from public archives."""
    
    # NASA Exoplanet Archive URLs (TAP service - current as of 2024)
    # Using the TAP (Table Access Protocol) service which is the current standard
    KOI_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+koi+where+koi_disposition+is+not+null&format=csv"
    TOI_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+toi&format=csv"
    K2_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+k2candidates&format=csv"
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the dataset downloader.
        
        Args:
            data_dir: Directory to save downloaded datasets
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def _download_file(self, url: str, filename: str) -> str:
        """
        Download a file from URL and save to data directory.
        
        Args:
            url: URL to download from
            filename: Name to save the file as
            
        Returns:
            Path to the downloaded file
        """
        filepath = os.path.join(self.data_dir, filename)
        
        print(f"Downloading {filename}...")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        print(f"Downloaded {filename} to {filepath}")
        return filepath
    
    def download_koi_dataset(self) -> str:
        """
        Download Kepler Objects of Interest (KOI) dataset.
        
        Returns:
            Path to the downloaded KOI dataset
        """
        return self._download_file(self.KOI_URL, "koi_dataset.csv")
    
    def download_toi_dataset(self) -> str:
        """
        Download TESS Objects of Interest (TOI) dataset.
        
        Returns:
            Path to the downloaded TOI dataset
        """
        return self._download_file(self.TOI_URL, "toi_dataset.csv")
    
    def download_k2_dataset(self) -> str:
        """
        Download K2 mission dataset.
        
        Returns:
            Path to the downloaded K2 dataset
        """
        return self._download_file(self.K2_URL, "k2_dataset.csv")
    
    def download_all_datasets(self) -> dict:
        """
        Download all available NASA exoplanet datasets.
        
        Returns:
            Dictionary mapping dataset names to file paths
        """
        datasets = {}
        
        try:
            datasets['koi'] = self.download_koi_dataset()
        except Exception as e:
            print(f"Failed to download KOI dataset: {e}")
        
        try:
            datasets['toi'] = self.download_toi_dataset()
        except Exception as e:
            print(f"Failed to download TOI dataset: {e}")
        
        try:
            datasets['k2'] = self.download_k2_dataset()
        except Exception as e:
            print(f"Failed to download K2 dataset: {e}")
        
        return datasets
