# 🏠 Melbourne House Price Prediction

A machine learning project to predict residential property prices in Melbourne, Australia. The pipeline covers end-to-end data wrangling, feature engineering, and model training — comparing Linear Regression, Random Forest, and XGBoost regressors.

---

## 📁 Project Structure

```
├── Raw_data.csv           # Original Melbourne housing dataset (~13,580 records)
├── cleaned_data.csv       # Processed & encoded dataset ready for modeling
├── data_wrangling.py      # Data cleaning, imputation, and feature engineering
├── Model_V1.py            # Baseline model comparison (LR, RF, XGBoost)
└── Model_V2.py            # XGBoost with hyperparameter tuning & feature importance
```

---

## 📊 Dataset

The raw dataset contains **13,580 Melbourne property listings** with 21 features including:

| Feature | Description |
|---|---|
| `Suburb` | Property suburb |
| `Rooms` | Number of rooms |
| `Type` | Property type (house, unit, townhouse) |
| `Price` | Sale price (target variable) |
| `Distance` | Distance from Melbourne CBD (km) |
| `Bathroom`, `Car` | Number of bathrooms / car spaces |
| `Landsize`, `BuildingArea` | Property dimensions (m²) |
| `YearBuilt` | Year of construction |
| `CouncilArea`, `Regionname` | Administrative region |

---

## 🔧 Pipeline Overview

### 1. Data Wrangling (`data_wrangling.py`)

**Missing value imputation:**
- `Car` → median fill
- `CouncilArea` → suburb-level mode mapping, fallback to global mode
- `YearBuilt` → suburb-level median, fallback to global median
- `BuildingArea` → median grouped by `Type` + `Rooms`, fallback to global median
- Missing indicator flags added for `BuildingArea` and `YearBuilt` (useful for tree models)

**Feature engineering:**
- `Age` = `2025 − YearBuilt` (clipped to 0–300)
- Dropped redundant/noisy columns: `Address`, `SellerG`, `Postcode`, `Bedroom2`, `Date`, `Lattitude`, `Longtitude`

**Encoding:**
- `Suburb` (high cardinality) → Target Encoding
- `Type`, `Method`, `CouncilArea`, `Regionname` (low cardinality) → One-Hot Encoding (drop-first)
- Numerical features → passed through as-is

---

### 2. Baseline Modeling (`Model_V1.py`)

Three models evaluated on an 80/20 train-validation split with **log-transformed target** (`log1p(Price)`):

- **Linear Regression** (with StandardScaler)
- **Random Forest** (100 trees, max depth 10)
- **XGBoost** (200 estimators, lr=0.1, max depth 6)

Metrics reported: MAE, RMSE, R², and 5-fold cross-validation MAE.

---

### 3. Tuned XGBoost (`Model_V2.py`)

Builds on V1 with `RandomizedSearchCV` (20 iterations, 3-fold CV) tuning over:

```
n_estimators, learning_rate, max_depth,
subsample, colsample_bytree, min_child_weight, gamma
```

Also generates a **top-15 feature importance chart** from the best estimator.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn xgboost category_encoders matplotlib seaborn
```

### Run the pipeline

```bash
# Step 1 — clean and encode the raw data
python data_wrangling.py

# Step 2 — train and compare baseline models
python Model_V1.py

# Step 3 — tune XGBoost and inspect feature importances
python Model_V2.py
```

> Make sure `Raw_data.csv` is in the same directory before running `data_wrangling.py`. The script will produce `cleaned_data.csv`, which is required by both model scripts.

---

## 📈 Model Comparison

| Model | Val MAE | Val RMSE | Val R² |
|---|---|---|---|
| Linear Regression | $230,979 | $1,215,847 | -2.7216 |
| Random Forest | $173,454 | $283,223 | 0.7981 |
| XGBoost (baseline) | $154,121 | $248,471 | 0.8446 |
| XGBoost (tuned) | $156,618 | $254,173 | 0.8374 |

> Fill in after running the scripts.

---

## 🔍 Key Design Decisions

- **Log-transformed target:** House prices are right-skewed; `log1p(Price)` stabilises variance and improves model fit. Predictions are inverse-transformed with `expm1` for reporting.
- **Missing indicators:** Binary flags for imputed `BuildingArea` and `YearBuilt` let tree models learn if missingness itself is informative.
- **Target encoding for `Suburb`:** With hundreds of suburbs, one-hot encoding would create excessive dimensionality. Target encoding maps each suburb to its mean target value, conditioned on the training set.
- **Group-level imputation:** Using suburb/type/room group medians preserves local pricing signals rather than collapsing to global statistics.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `scikit-learn` | Preprocessing, modeling, evaluation |
| `xgboost` | Gradient boosting regressor |
| `category_encoders` | Target encoding for high-cardinality categoricals |
| `matplotlib`, `seaborn` | Visualisation |
