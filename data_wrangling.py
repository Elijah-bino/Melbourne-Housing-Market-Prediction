import pandas as pd
import numpy as np
FILE_PATH = 'Raw_data.csv'
df = pd.read_csv(FILE_PATH)

# ── 1. Quick check: only show columns that actually have missing values
missing = df.isnull().sum()
missing = missing[missing > 0]  # filter to only columns with at least 1 missing

if len(missing) == 0:
    print("No missing values in the entire dataset!")
else:
    # Create a nice little table-like output
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': (missing / len(df) * 100).round(2)
    }).sort_values('Missing Count', ascending=False)
    
    print("Columns with missing values:")
    print("═══════════════════════════")
    print(missing_df)
    print(f"\nTotal missing cells: {missing.sum():,}")
    print(f"Number of affected columns: {len(missing)}")

# ── 2. Imputation ───────────────────────────────────────────────────

# Car: simple median - very few missing
df['Car'] = df['Car'].fillna(df['Car'].median())

# CouncilArea: best-effort mapping from Suburb → mode per suburb
suburb_council_mode = df.groupby('Suburb')['CouncilArea'].agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
df['CouncilArea'] = df.apply(
    lambda row: suburb_council_mode[row['Suburb']] if pd.isna(row['CouncilArea']) else row['CouncilArea'],
    axis=1
)
# Still any left? (very rare) → fill with overall mode
df['CouncilArea'] = df['CouncilArea'].fillna(df['CouncilArea'].mode()[0])

# YearBuilt: median per Suburb
df['YearBuilt'] = df.groupby('Suburb')['YearBuilt'].transform(lambda x: x.fillna(x.median()))
# Remaining (suburbs with all missing YearBuilt) → global median
df['YearBuilt'] = df['YearBuilt'].fillna(df['YearBuilt'].median())

# BuildingArea: median per Type + Rooms (most similar properties)
df['BuildingArea'] = df.groupby(['Type', 'Rooms'])['BuildingArea'].transform(lambda x: x.fillna(x.median()))
# Still remaining? → global median
df['BuildingArea'] = df['BuildingArea'].fillna(df['BuildingArea'].median())

# ── 3. Add missing indicators (very useful for tree models like XGBoost/RF)
df['BuildingArea_was_missing'] = df['BuildingArea'].isna().astype(int)
df['YearBuilt_was_missing']    = df['YearBuilt'].isna().astype(int)

# ── 4. After snapshot ──────────────────────────────────────────────
print("\nMissing values AFTER imputation:")
print(df[['Car', 'CouncilArea', 'YearBuilt', 'BuildingArea']].isnull().sum())

print("\nQuick stats after imputation:")
print(df[['YearBuilt', 'BuildingArea']].describe().round(1))



# Quick inspection after imputation

print("Shape of the data now:", df.shape)
print("\nMissing values ANYWHERE in the dataset after imputation?")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\n── Key columns summary after imputation ──")
print(df[['Rooms', 'Type', 'Price', 'Distance', 'Bedroom2', 'Bathroom',
          'Car', 'Landsize', 'BuildingArea', 'YearBuilt', 'CouncilArea',
          'Regionname', 'BuildingArea_was_missing', 'YearBuilt_was_missing']]
      .describe(include='all').round(2))

print("\nMedian values after imputation:")
print(df[['YearBuilt', 'BuildingArea', 'Car']].median())

print("\nHow many rows still have missing values in any column?")
print("Rows with at least one NaN:", df.isnull().any(axis=1).sum())

print("\nUnique values in important categoricals:")
print("Suburb:", df['Suburb'].nunique())
print("CouncilArea:", df['CouncilArea'].nunique())
print("Regionname:", df['Regionname'].nunique())
print("Type:", df['Type'].unique())

# Drop unwanted columns
cols_to_drop = [
    'Address',           # unique, useless
    'SellerG',           # high cardinality noise / potential leak
    'Postcode',          # redundant with Suburb/Region/Distance
    'Bedroom2',          # redundant with Rooms
    'Date',              # use extracted year/month instead if needed
    'Lattitude',
    'Longtitude',
]

df = df.drop(columns=cols_to_drop)

# ── 1. Create Age feature ────────────────────────────────────────
CURRENT_YEAR = 2025  # or 2024 if you prefer; or use datetime.now().year

# Check if YearBuilt exists and has reasonable values
if 'YearBuilt' in df.columns:
    print("\nYearBuilt stats:")
    print(df['YearBuilt'].describe())
    
    # Create Age (handle any implausible values)
    df['Age'] = CURRENT_YEAR - df['YearBuilt']
    
    # Optional: clip unrealistic ages (e.g., negative or >300 years)
    df['Age'] = df['Age'].clip(lower=0, upper=300)
    
    print("\nAge stats after creation:")
    print(df['Age'].describe())
    
    # Quick correlation check with log(Price)
    if 'Price' in df.columns:
        print("\nCorrelation between Age and log(Price):",
              np.corrcoef(df['Age'], np.log1p(df['Price']))[0,1].round(4))
else:
    print("Error: 'YearBuilt' column not found!")

# ── 2. Reorder columns (optional — put Age near YearBuilt) ───────
cols = list(df.columns)
if 'Age' in cols and 'YearBuilt' in cols:
    year_idx = cols.index('YearBuilt')
    cols.pop(cols.index('Age'))
    cols.insert(year_idx + 1, 'Age')
    df = df[cols]

print(df.columns)


# ────────────────────────────────────────────────────────────────
#          ENCODING CATEGORICAL VARIABLES – FIXED VERSION
# ────────────────────────────────────────────────────────────────

from category_encoders import TargetEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

print("Shape before encoding:", df.shape)

# Define columns
high_card_cat = ['Suburb']
low_card_cat  = ['Type', 'Method', 'CouncilArea', 'Regionname']
numeric_cols  = ['Rooms', 'Distance', 'Bathroom', 'Car', 'Landsize', 
                 'BuildingArea', 'YearBuilt', 'Propertycount', 
                 'BuildingArea_was_missing', 'YearBuilt_was_missing',]

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('target_enc', TargetEncoder(), high_card_cat),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'), low_card_cat),
        ('pass', 'passthrough', numeric_cols)
    ],
    remainder='drop'
)

# Separate X and y
X = df.drop('Price', axis=1)
y = df['Price']  # keep original scale for now

# Fit & transform X using y for TargetEncoder
encoded_X = preprocessor.fit_transform(X, y)

# Get column names
onehot_features = preprocessor.named_transformers_['onehot'].get_feature_names_out(low_card_cat)
target_features = high_card_cat
pass_features   = numeric_cols

all_features = list(onehot_features) + target_features + pass_features

# Create encoded DataFrame (features only)
df_encoded = pd.DataFrame(encoded_X, columns=all_features)

# Add back the target
df_encoded['Price'] = y.values

print("\nShape after encoding:", df_encoded.shape)
print("\nColumns after encoding:")
print(df_encoded.columns.tolist())

# Quick peek
print("\nFirst 3 rows:")
print(df_encoded.head(3))

# Save
output_file = "cleaned_data.csv"
df_encoded.to_csv(output_file, index=False)
print(f"\nSaved to {output_file}")