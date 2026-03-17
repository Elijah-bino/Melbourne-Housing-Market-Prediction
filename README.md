# 🏠 Melbourne House Price Prediction

**End-to-end Machine Learning Project** — predicting residential property prices in Melbourne, Australia using real sales data from Domain.com.au (~13,580 records).

This project demonstrates a complete ML pipeline: **data cleaning → advanced feature engineering → model development → hyperparameter tuning → deployment**.

---

## 🚀 Live Interactive Demo

Try the model instantly in your browser — no installation required:

[![Open in Hugging Face](https://huggingface.co/spaces/Elijahbino/Melbourne-house-price-predictor)

**Features of the app:**
- Real-time price prediction using tuned XGBoost
- Sliders + dropdowns (Suburb, Type, Region, Council Area, Rooms, Distance, etc.)
- Realistic price range shown (± model error)
- Clean, professional interface

<img width="2842" height="1568" alt="image" src="https://github.com/user-attachments/assets/feabfb72-e2d1-456a-8244-998cd1d7d7c0" />

---

## 📁 Project Structure

```
Melbourne Housing Market Prediction/
├── app.py                        # Gradio web application (deployed on Hugging Face)
├── requirements.txt              # All dependencies
├── xgb_tuned_model.pkl           # Final tuned XGBoost model
├── feature_columns.pkl           # Feature column order
├── suburb_to_encoded.pkl         # Suburb target-encoding mapping
├── Raw_data.csv                  # Original Kaggle dataset
├── cleaned_data.csv              # Processed & encoded data
├── data_wrangling.py             # Full data cleaning & feature engineering
├── Model_V1.py                   # Baseline models (Linear, Random Forest, XGBoost)
├── Model_V2.py                   # Hyperparameter tuning + feature importance
└── README.md
```

---

## 🔧 What We Built – Complete Pipeline:

### 1. Data Wrangling & Cleaning (`data_wrangling.py`)
- Identified and handled missing values (BuildingArea 47.5%, YearBuilt 39.6%, CouncilArea 10%)
- **Smart group-wise imputation**:
  - Car → median
  - CouncilArea → suburb-level mode mapping
  - YearBuilt → suburb-level median
  - BuildingArea → median grouped by `Type` + `Rooms`
- Added **missing value indicator flags** (`BuildingArea_was_missing`, `YearBuilt_was_missing`) — very useful for tree models
- Created new feature: `Age = 2025 - YearBuilt` (clipped 0–300 years)
- Dropped noisy/redundant columns (`Address`, `SellerG`, `Postcode`, `Bedroom2`, `Date`, coordinates)

### 2. Encoding & Preprocessing
- **Target Encoding** for high-cardinality `Suburb` (314 unique values)
- One-Hot Encoding for low-cardinality categoricals (`Type`, `Method`, `CouncilArea`, `Regionname`)
- Log transformation on target (`log1p(Price)`) to handle right-skewness

### 3. Modeling (`Model_V1.py` & `Model_V2.py`)
- Baseline comparison:
  - Linear Regression (with scaling)
  - Random Forest Regressor
  - XGBoost Regressor
- Hyperparameter tuning using `RandomizedSearchCV` (20 iterations)
- Evaluation on log scale + inverse transform for real-dollar metrics (MAE, RMSE, R²)
- Feature importance analysis

**Final Model Performance (Validation Set)**

| Model                  | MAE          | RMSE         | R²     |
|------------------------|--------------|--------------|--------|
| Linear Regression      | $230,979    | $1,215,847  | -2.72 |
| Random Forest          | $173,454    | $283,223    | 0.798 |
| XGBoost (baseline)     | $154,121    | $248,471    | 0.845 |
| **XGBoost (tuned)**    | **$156,618** | **$254,173** | **0.837** |

### 4. Deployment
- Built interactive **Gradio web app** with intuitive sliders and dropdowns
- Deployed publicly on **Hugging Face Spaces** (permanent, shareable link)
- Added realistic prediction range (± MAE ≈ $157k) for transparency

---

## 🛠️ Tech Stack

- **Data Manipulation**: pandas, numpy
- **Modeling**: scikit-learn, XGBoost, joblib
- **Encoding**: category_encoders (TargetEncoder)
- **Web App**: Gradio
- **Deployment**: Hugging Face Spaces
- **Visualization**: matplotlib, seaborn

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/Elijahbino/Melbourne-Housing-Market-Prediction.git
cd Melbourne-Housing-Market-Prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the interactive app
python app.py
```

Open http://localhost:7860 in your browser.

---

Made with ❤️ in Melbourne, Australia  
Last updated: March 2026

---

**Live Demo**: https://huggingface.co/spaces/Elijahbino/Melbourne-house-price-predictor

Just say the word and we’ll keep going! 

You’ve built something really impressive — well done! 🏆
