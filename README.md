# Genre Mapper
A lightweight app that predicts a song’s genre from a YouTube link.

Users paste a link, and the app downloads a short audio segment, extracts acoustic features, scales them using a trained model, and displays the predicted genre.

# Video Demo

# How It's Made 
Tech used: Python, FastAPI, NumPy/SciPy, scikit-learn, librosa, ffmpeg, yt-dlp, HTML/CSS (dark), Docker, Fly.io

  - Dataset & Genres: GTZAN (10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock).
  - Features (per clip): MFCCs (20 × mean/std), Chroma (mean/std), spectral centroid/bandwidth/rolloff/contrast (mean/std), RMS (mean/std), ZCR (mean/std), tempo → ~89 dims.
  - Scaling: StandardScaler trained on Train only; reused for Val/Test/Inference.
  - Model: Linear SVM (multiclass). Hyperparam C tuned on Validation by Macro-F1.
  - Metrics (Test): Accuracy ≈ 0.73, Macro-F1 ≈ 0.73 (recorded in artifacts/metrics.txt).
  - Inference flow: validate URL → download audio → convert mono @ 22.05 kHz s16 → take middle 30s → extract features (locked order) → scale → predict → render Top-1/Top-3.

# Lessons Learned 

  - Normalization matters: Standardized features to stabilize linear models and make coefficients comparable.
  - Feature engineering: Started with classic, interpretable DSP features (MFCCs, Chroma, spectral stats, RMS, ZCR, tempo); aggregated via mean/std to get fixed-length vectors.
  - Baselines first: Chose Linear SVM as a strong, simple baseline; tuned only C with Validation to avoid overfitting.
  - Reproducibility: Saved artifacts (scaler, model, feature order, label map, metrics) and scripted the pipeline end-to-end.
  - Practical data issues: Handled corrupt/unsupported audio gracefully (skip + log) and standardized the audio format.

# Quick Start 

Inside the genre mapper directory 
  ### 0) Python env
    python -m venv .venv 
  
    Windows: .\.venv\Scripts\Activate.ps1 
  
    macOS/Linux: source .venv/bin/activate 
  
    pip install --upgrade pip
  
    pip install -r requirements.txt

  ### 1) Generate GTZAN metadata and splits
  Download the GTZAN data set
  
    python scripts/make_metadata_gtzan.py
  
    python scripts/validate_metadata.py
  
    python scripts/split_dataset.py --meta metadata.csv --out splits --seed 42

  ### 2) Feature extraction
  
    python scripts/extract_features.py --csv splits/train.csv --out artifacts/train
    
    python scripts/extract_features.py --csv splits/val.csv   --out artifacts/val
    
    python scripts/extract_features.py --csv splits/test.csv  --out artifacts/test

  ### 3) Fit scaler + train model + sanity check
  
    python scripts/fit_scaler.py
  
    python scripts/train_model.py
  
    python scripts/check_artifacts.py

  ### 4) Run the app locally
  
    uvicorn app.api:app --reload

# File Structure 

<details>
  <summary><b>File Structure</b></summary>

```text
genre-mapper/
├─ app/
│  ├─ api.py               
│  └─ infer_core.py        
├─ artifacts/
│  ├─ model.pkl           
│  ├─ scaler.pkl           
│  ├─ feature_order.json   
│  ├─ label_map.json       
│  └─ metrics.txt         
├─ data/
│  └─ gtzan/               # GTZAN 30s clips (not in repo)
├─ scripts/
│  ├─ make_metadata_gtzan.py
│  ├─ validate_metadata.py
│  ├─ split_dataset.py
│  ├─ extract_features.py
│  ├─ fit_scaler.py
│  ├─ train_model.py
│  ├─ check_artifacts.py
│  └─ youtube_fetch_clip.py # add 30s clips by genre to expand dataset
├─ splits/
│  ├─ train.csv
│  ├─ val.csv
│  └─ test.csv
├─ requirements.txt
└─ metadata.csv
