import gradio as gr
import pandas as pd
import joblib
import numpy as np
import os

# ── Paths (adjust if needed) ────────────────────────────────────────────────
RAW_DATA_PATH = "Raw_data.csv"          # or "cleaned_data.csv"
MODEL_PATH = "xgb_tuned_model.pkl"
FEATURE_COLS_PATH = "feature_columns.pkl"
suburb_to_encoded = joblib.load("suburb_to_encoded.pkl")

# ── Load model & expected feature columns ───────────────────────────────────
model = joblib.load(MODEL_PATH)
feature_cols = joblib.load(FEATURE_COLS_PATH)

# ── Load data to get real dropdown options ──────────────────────────────────
if not os.path.exists(RAW_DATA_PATH):
    raise FileNotFoundError(f"Cannot find {RAW_DATA_PATH}. Please place it in the same folder as app.py")

df_raw = pd.read_csv(RAW_DATA_PATH)

# Get sorted unique values for dropdowns
TYPE_OPTIONS = sorted(df_raw['Type'].dropna().unique().tolist())               # e.g. ['h', 't', 'u']
REGION_OPTIONS = sorted(df_raw['Regionname'].dropna().unique().tolist())       # 8 regions
COUNCIL_OPTIONS = sorted(df_raw['CouncilArea'].dropna().unique().tolist())     # ~33 councils
SUBURB_OPTIONS = sorted(df_raw['Suburb'].dropna().unique().tolist())           # ~314 suburbs

print(f"Loaded {len(SUBURB_OPTIONS)} suburbs, {len(REGION_OPTIONS)} regions, "
      f"{len(COUNCIL_OPTIONS)} councils, {len(TYPE_OPTIONS)} types")

# ── Prediction function ─────────────────────────────────────────────────────
def predict_price(
    rooms, distance, bathroom, car, landsize, building_area, age,
    property_type, region, council_area, suburb
):
    if suburb not in SUBURB_OPTIONS:
        return "Error: Selected suburb not found in training data."

    # Build input dictionary (same as before)
    input_dict = {
        'Rooms': rooms,
        'Distance': distance,
        'Bathroom': bathroom,
        'Car': car,
        'Landsize': landsize,
        'BuildingArea': building_area,
        'Age': age,
    }

    # One-hot: Type
    for t in TYPE_OPTIONS:
        input_dict[f'Type_{t}'] = 1 if property_type == t else 0

    # One-hot: Regionname
    for r in REGION_OPTIONS:
        col = f'Regionname_{r}'
        if col in feature_cols:
            input_dict[col] = 1 if region == r else 0

    # One-hot: CouncilArea
    for c in COUNCIL_OPTIONS:
        col = f'CouncilArea_{c}'
        if col in feature_cols:
            input_dict[col] = 1 if council_area == c else 0

    # Suburb encoded value (using the mapping you saved)
    suburb_encoded = suburb_to_encoded.get(suburb, suburb_to_encoded.get("Abbotsford", 1000000))
    input_dict['Suburb'] = suburb_encoded

    # Fill any missing columns with 0
    for col in feature_cols:
        if col not in input_dict:
            input_dict[col] = 0.0

    # Create DataFrame in correct order
    input_df = pd.DataFrame([input_dict])[feature_cols]

    # Predict
    pred_log = model.predict(input_df)[0]
    pred_price = np.expm1(pred_log)

    # Realistic range based on validation MAE
    mae = 156618
    lower = max(0, pred_price - mae)
    upper = pred_price + mae

    # Nice formatted output
    output_text = f"""
**Estimated Price:** ${pred_price:,.0f} AUD

**Realistic Range** (based on model error):  
${lower:,.0f} – ${upper:,.0f}

This is an estimate from an XGBoost model trained on Melbourne sales data.  
Actual sale prices can vary due to market conditions, negotiation, etc.  
(Model performance on validation set: MAE ≈ $157k, R² = 0.84)
"""

    return output_text

# ── Gradio Interface ────────────────────────────────────────────────────────
with gr.Blocks(title="Melbourne House Price Predictor") as demo:
    gr.Markdown("# 🏠 Melbourne House Price Predictor (XGBoost)")
    gr.Markdown("Choose values below → get instant prediction based on real Melbourne data")

    with gr.Row():
        with gr.Column():
            rooms = gr.Slider(1, 10, value=3, step=1, label="Rooms")
            distance = gr.Slider(0, 50, value=10, step=0.5, label="Distance to CBD (km)")
            bathroom = gr.Slider(0, 8, value=1, step=1, label="Bathrooms")
            car = gr.Slider(0, 10, value=1, step=1, label="Car spaces")
            landsize = gr.Slider(0, 5000, value=500, step=10, label="Landsize (m²)")
            building_area = gr.Slider(0, 1000, value=120, step=5, label="Building Area (m²)")
            age = gr.Slider(0, 300, value=40, step=1, label="Age of house (years)")

            property_type = gr.Dropdown(
                choices=TYPE_OPTIONS, value="h", label="Property Type"
            )
            region = gr.Dropdown(
                choices=REGION_OPTIONS, value="Southern Metropolitan", label="Region"
            )
            council = gr.Dropdown(
                choices=COUNCIL_OPTIONS, value="Yarra", label="Council Area"
            )
            suburb = gr.Dropdown(
                choices=SUBURB_OPTIONS,
                value="Abbotsford",
                label="Suburb",
                interactive=True,
                allow_custom_value=False
            )

        with gr.Column():
            output = gr.Markdown("**Prediction will appear here**")

    submit_btn = gr.Button("Predict Price", variant="primary")
    submit_btn.click(
        fn=predict_price,
        inputs=[rooms, distance, bathroom, car, landsize, building_area, age,
                property_type, region, council, suburb],
        outputs=output
    )

    gr.Examples(
        examples=[
            [3, 10, 1, 1, 500, 120, 50, "h", "Southern Metropolitan", "Yarra", "Abbotsford"],
            [4, 15, 2, 2, 700, 180, 30, "t", "Eastern Metropolitan", "Monash", "Glen Waverley"]
        ],
        inputs=[rooms, distance, bathroom, car, landsize, building_area, age,
                property_type, region, council, suburb],
        label="Quick Examples"
    )

demo.launch(server_name="0.0.0.0", server_port=7860)