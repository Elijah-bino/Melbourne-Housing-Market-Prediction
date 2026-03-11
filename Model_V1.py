# ────────────────────────────────────────────────────────────────
#          MELBOURNE HOUSE PRICE PREDICTION – MODELING
# ────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── 1. Load the cleaned data ─────────────────────────────────────
df = pd.read_csv("cleaned_data.csv")

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nFirst 3 rows:")
print(df.head(3))

# ── 2. Separate features and target ──────────────────────────────
X = df.drop("Price", axis=1)
y = df["Price"]

# Optional: use log-transformed target (strongly recommended for house prices)
y_log = np.log1p(y)   # log(Price + 1) to handle any very small prices
# We'll try both and compare

# ── 3. Train / validation split ──────────────────────────────────
# For simplicity: random 80/20 split
# Later we can do time-based using original Date if you kept it

X_train, X_val, y_train, y_val = train_test_split(
    X, y_log,
    test_size=0.2,
    random_state=42
)

print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

# ── 4. Define evaluation function ────────────────────────────────
def evaluate_model(model, X_tr, y_tr, X_vl, y_vl, name="Model"):
    # Fit on train
    model.fit(X_tr, y_tr)
    
    # Predict on validation
    pred = model.predict(X_vl)
    
    pred_original = np.expm1(pred)  # inverse of log1p
    y_vl_original = np.expm1(y_vl)

    mae = mean_absolute_error(y_vl_original, pred_original)
    rmse = np.sqrt(mean_squared_error(y_vl_original, pred_original))
    r2 = r2_score(y_vl_original, pred_original)
    
    print(f"\n{name} Performance on Validation Set:")
    print(f"MAE:  ${mae:,.0f}")
    print(f"RMSE: ${rmse:,.0f}")
    print(f"R²:   {r2:.4f}")
    
    # Also do 5-fold CV on train for more robust estimate
    cv_mae = -cross_val_score(model, X_tr, y_tr, cv=5, scoring='neg_mean_absolute_error').mean()
    print(f"5-fold CV MAE (on train): ${cv_mae:,.0f}")

# ── 5. Baseline: Linear Regression ───────────────────────────────
print("\n=== Linear Regression (Baseline) ===")
lr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

evaluate_model(lr_pipe, X_train, y_train, X_val, y_val, "Linear Regression")

# ── 6. Random Forest ─────────────────────────────────────────────
print("\n=== Random Forest Regressor ===")
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

evaluate_model(rf, X_train, y_train, X_val, y_val, "Random Forest")

# ── 7. XGBoost ───────────────────────────────────────────────────
print("\n=== XGBoost Regressor ===")
xgb = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)

evaluate_model(xgb, X_train, y_train, X_val, y_val, "XGBoost")