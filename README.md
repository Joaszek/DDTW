# DTW Cost Classification

A machine learning pipeline for classifying biomedical signals using Dynamic Time Warping (DTW) cost features.

## Project Structure

```
DDTW/
├── config.yaml              # Configuration file with all parameters
├── config.py                # Configuration loading module
├── data_utils.py            # Data loading and preprocessing utilities
├── feature_engineering.py   # Feature extraction functions
├── data_pipeline.py         # Data pipeline operations
├── train.py                 # Main training script
├── requirements.txt         # Python dependencies
├── cost_dtw_v2_model.py    # Original monolithic script (legacy)
└── ml-data/                 # Data directory
    ├── BAS/
    ├── B6/
    ├── B10/
    └── B15/
```

## Module Descriptions

### `config.py`
Configuration management module that loads parameters from `config.yaml`:
- Data paths and categories
- Parallel file patterns
- Model hyperparameters
- Preprocessing flags

### `data_utils.py`
Core utilities for data handling:
- `load_cost_vector()` - Load cost vectors from CSV files
- `find_parallel_file()` - Find matching parallel signal files
- `preprocess_cost_vector()` - Apply log transform and normalization
- `resample_1d()` - Resample arrays to target length

### `feature_engineering.py`
Feature extraction functions:
- `window_stats()` - Calculate statistical features from windows
- `corr_safe()` - Safe correlation calculation
- `build_window_features()` - Build comprehensive feature vectors

### `data_pipeline.py`
High-level data pipeline operations:
- `load_category_data()` - Load and process single category
- `load_all_data()` - Load and combine all categories
- `split_train_test()` - Split data into train/test sets

### `train.py`
Main training script with:
- Model definitions (LogReg, SVM, RandomForest, HistGradientBoosting, CatBoost)
- Training and evaluation pipeline
- Visualization of confusion matrices

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your data is organized in the `ml-data/` directory with subdirectories for each category.

## Usage

### Basic Training

Run the training pipeline:
```bash
python train.py
```

This will:
1. Load data from configured categories
2. Extract features using sliding windows
3. Train multiple models
4. Display accuracy metrics and confusion matrices

### Configuration

Edit `config.yaml` to modify:

```yaml
# Data configuration
data_root: "./ml-data"
categories:
  - "BAS"
  - "B6"

# Window parameters
window_len: 32
window_step: 8

# Preprocessing
use_log1p: true
use_norm_by_median: true
```

### Using Individual Modules

```python
from config import config
from data_pipeline import load_all_data, split_train_test

# Load data
X, y, meta, all_rows = load_all_data(categories=["BAS", "B6"])

# Split train/test
X_train, y_train, X_test, y_test = split_train_test(all_rows, X, y)
```

## Models

The pipeline trains and evaluates multiple models:

1. **Logistic Regression** - Baseline linear model
2. **SVM-RBF** - Support Vector Machine with RBF kernel
3. **Random Forest** - Ensemble of decision trees
4. **Histogram Gradient Boosting** - Efficient gradient boosting
5. **CatBoost** (optional) - Gradient boosting with categorical support

## Features

The feature engineering pipeline extracts:
- Statistical features per channel (mean, std, percentiles, etc.)
- Cross-channel relationships (differences, ratios, correlations)
- Shape features via resampling and normalization

## Development

### Adding New Models

Edit `train.py` and add to the `get_models()` function:

```python
def get_models():
    return {
        "YourModel": YourModelClass(param1=value1, ...),
        ...
    }
```

### Modifying Features

Edit `feature_engineering.py` to add or modify feature extraction logic in `build_window_features()`.

### Changing Data Processing

Modify preprocessing in `data_utils.py` or adjust pipeline logic in `data_pipeline.py`.
