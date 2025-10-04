'use client';

import { useState, useRef } from 'react';
import { classifyBatch, handleApiError } from '@/lib/api';
import type { ExoplanetFeatures, BatchClassificationResponse } from '@/lib/types';

interface FileUploadProps {
  onResults: (results: BatchClassificationResponse) => void;
  onLoadingChange: (loading: boolean) => void;
}

export default function FileUpload({ onResults, onLoadingChange }: FileUploadProps) {
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFileSelect(e.target.files[0]);
    }
  };

  const handleFileSelect = (selectedFile: File) => {
    if (!selectedFile.name.endsWith('.csv')) {
      setError('Please upload a CSV file');
      return;
    }
    setFile(selectedFile);
    setError(null);
  };

  const parseCSV = (text: string): ExoplanetFeatures[] => {
    const lines = text.trim().split('\n');
    const headers = lines[0].split(',').map(h => h.trim());
    
    // Validate headers
    const requiredHeaders = ['orbital_period', 'transit_duration', 'transit_depth', 'planetary_radius'];
    const missingHeaders = requiredHeaders.filter(h => !headers.includes(h));
    if (missingHeaders.length > 0) {
      throw new Error(`Missing required columns: ${missingHeaders.join(', ')}`);
    }

    const observations: ExoplanetFeatures[] = [];
    
    for (let i = 1; i < lines.length; i++) {
      if (!lines[i].trim()) continue;
      
      const values = lines[i].split(',').map(v => v.trim());
      const observation: any = {};
      
      headers.forEach((header, index) => {
        const value = values[index];
        if (value && value !== '') {
          observation[header] = parseFloat(value);
        }
      });

      // Validate required fields
      if (
        observation.orbital_period &&
        observation.transit_duration &&
        observation.transit_depth &&
        observation.planetary_radius
      ) {
        observations.push(observation as ExoplanetFeatures);
      }
    }

    if (observations.length === 0) {
      throw new Error('No valid observations found in CSV');
    }

    if (observations.length > 1000) {
      throw new Error('Maximum 1000 observations allowed per batch');
    }

    return observations;
  };

  const handleSubmit = async () => {
    if (!file) {
      setError('Please select a file first');
      return;
    }

    setError(null);
    onLoadingChange(true);

    try {
      const text = await file.text();
      const observations = parseCSV(text);
      
      const results = await classifyBatch({ observations });
      onResults(results);
    } catch (err) {
      const errorMessage = handleApiError(err);
      setError(errorMessage);
    } finally {
      onLoadingChange(false);
    }
  };

  return (
    <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
      <h3 className="text-xl font-semibold text-white mb-6">Upload CSV File</h3>

      {/* Drag and Drop Area */}
      <div
        className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
          dragActive
            ? 'border-blue-500 bg-blue-900/20'
            : 'border-gray-600 hover:border-gray-500'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          onChange={handleChange}
          className="hidden"
        />

        {file ? (
          <div className="space-y-4">
            <div className="text-5xl">📄</div>
            <div>
              <div className="text-white font-semibold">{file.name}</div>
              <div className="text-sm text-gray-400 mt-1">
                {(file.size / 1024).toFixed(2)} KB
              </div>
            </div>
            <button
              onClick={() => {
                setFile(null);
                if (fileInputRef.current) {
                  fileInputRef.current.value = '';
                }
              }}
              className="text-red-400 hover:text-red-300 text-sm"
            >
              Remove file
            </button>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="text-5xl">📁</div>
            <div>
              <p className="text-gray-300 mb-2">
                Drag and drop your CSV file here
              </p>
              <p className="text-gray-500 text-sm mb-4">or</p>
              <button
                onClick={() => fileInputRef.current?.click()}
                className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
              >
                Browse Files
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Error Message */}
      {error && (
        <div className="mt-4 bg-red-900/20 border border-red-700 rounded-lg p-4">
          <p className="text-red-400 text-sm">❌ {error}</p>
        </div>
      )}

      {/* Upload Button */}
      {file && (
        <button
          onClick={handleSubmit}
          className="w-full mt-6 px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all transform hover:scale-105 shadow-lg hover:shadow-blue-500/50"
        >
          🚀 Process CSV File
        </button>
      )}

      {/* File Requirements */}
      <div className="mt-6 text-sm text-gray-400">
        <p className="font-semibold text-gray-300 mb-2">Requirements:</p>
        <ul className="list-disc list-inside space-y-1">
          <li>File must be in CSV format</li>
          <li>Maximum 1,000 observations per file</li>
          <li>Required columns: orbital_period, transit_duration, transit_depth, planetary_radius</li>
          <li>Optional column: equilibrium_temperature</li>
        </ul>
      </div>
    </div>
  );
}
