import pandas as pd
import joblib
from category_encoders import TargetEncoder

df = pd.read_csv("Raw_data.csv")
encoder = TargetEncoder()
encoder.fit(df[['Suburb']], df['Price'])
suburb_to_encoded = dict(zip(df['Suburb'], encoder.transform(df[['Suburb']])['Suburb']))

joblib.dump(suburb_to_encoded, "suburb_to_encoded.pkl")
print("Saved suburb → encoded mapping!")