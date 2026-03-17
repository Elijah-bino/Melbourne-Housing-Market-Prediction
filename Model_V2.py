# ────────────────────────────────────────────────────────────────
#          MELBOURNE HOUSE PRICE PREDICTION – MODEL V2 (XGBoost + Tuning)
# ────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── 1. Load cleaned data ─────────────────────────────────────────
df = pd.read_csv("cleaned_data.csv")
print("Shape:", df.shape)

X = df.drop("Price", axis=1)
y = np.log1p(df["Price"])  # log target

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 2. Evaluation function (fixed for log scale) ─────────────────
def evaluate_model(model, X_tr, y_tr, X_vl, y_vl, name="XGBoost"):
    model.fit(X_tr, y_tr)
    pred_log = model.predict(X_vl)
    pred = np.expm1(pred_log)
    y_vl_orig = np.expm1(y_vl)

    mae = mean_absolute_error(y_vl_orig, pred)
    rmse = np.sqrt(mean_squared_error(y_vl_orig, pred))
    r2 = r2_score(y_vl_orig, pred)

    print(f"\n{name} Performance on Validation Set:")
    print(f"MAE:  ${mae:,.0f}")
    print(f"RMSE: ${rmse:,.0f}")
    print(f"R²:   {r2:.4f}")

    # Fixed CV: use log scale for scoring, then report
    cv_scores = cross_val_score(model, X_tr, y_tr, cv=5, scoring='neg_mean_absolute_error')
    cv_mae_log = -cv_scores.mean()
    print(f"5-fold CV MAE (log scale): {cv_mae_log:.4f} (log dollars)")

# ── 3. Baseline XGBoost (same as before) ─────────────────────────
print("=== Baseline XGBoost ===")
xgb_base = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)

evaluate_model(xgb_base, X_train, y_train, X_val, y_val, "Baseline XGBoost")

# ── 4. Hyperparameter tuning ─────────────────────────────────────
print("\n=== Tuning XGBoost with RandomizedSearchCV ===")

param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 6, 8, 10],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2]
}

xgb_tuned = RandomizedSearchCV(
    estimator=XGBRegressor(random_state=42, n_jobs=-1),
    param_distributions=param_dist,
    n_iter=20,          # 20 random combinations
    cv=3,               # 3-fold CV
    scoring='neg_mean_absolute_error',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

xgb_tuned.fit(X_train, y_train)

print("\nBest parameters:", xgb_tuned.best_params_)
print("Best CV score (neg MAE log):", xgb_tuned.best_score_)

# Evaluate tuned model
evaluate_model(xgb_tuned.best_estimator_, X_train, y_train, X_val, y_val, "Tuned XGBoost")

# ── 5. Feature importance (from tuned model) ─────────────────────
best_model = xgb_tuned.best_estimator_
importances = pd.Series(best_model.feature_importances_, index=X_train.columns)
top_features = importances.sort_values(ascending=False).head(15)

print("\nTop 15 Feature Importances:")
print(top_features)

import joblib
joblib.dump(xgb_tuned.best_estimator_, "xgb_tuned_model.pkl")
joblib.dump(list(X_train.columns), "feature_columns.pkl")   # saves the exact column order
print("✅ Model and columns saved!")

# Plot
plt.figure(figsize=(10, 6))
top_features.plot(kind='barh', color='skyblue')
plt.title("Top 15 Feature Importances (Tuned XGBoost)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()