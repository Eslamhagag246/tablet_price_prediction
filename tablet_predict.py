import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
df = pd.read_csv('final_project/tablets/tablets_cleaned_clean.csv')

# ─────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────

# Clean price: remove 'EGP' and commas → float
df['price'] = df['price'].str.replace('EGP', '', regex=False)\
                         .str.replace(',', '', regex=False)\
                         .str.strip()\
                         .astype(float)

# Normalize text columns to lowercase
df['brand']   = df['brand'].str.lower().str.strip()
df['website'] = df['website'].str.lower().str.strip()
df['stock']   = df['stock'].str.lower().str.strip()

# Parse timestamp → extract date features
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['year']      = df['timestamp'].dt.year
df['month']     = df['timestamp'].dt.month
df['day']       = df['timestamp'].dt.day
df['dayofweek'] = df['timestamp'].dt.dayofweek

# Fill missing RAM with median
df['ram_gb'] = df['ram_gb'].fillna(df['ram_gb'].median())

# Feature engineering
df['ram_storage_ratio'] = df['ram_gb'] / (df['storage_gb'] + 1)
df['storage_per_egp']   = df['storage_gb'] / df['price']

# Encode categorical columns — keep encoders to reuse on new data
le_brand   = LabelEncoder()
le_website = LabelEncoder()
df['brand_enc']   = le_brand.fit_transform(df['brand'])
df['website_enc'] = le_website.fit_transform(df['website'])

# ─────────────────────────────────────────
# 3. FEATURES & TARGET
# ─────────────────────────────────────────
FEATURES = [
    'ram_gb', 'storage_gb',
    'brand_enc', 'website_enc',
    'year', 'month', 'day', 'dayofweek',
    'ram_storage_ratio', 'storage_per_egp'
]

X = df[FEATURES]
y = df['price']

# ─────────────────────────────────────────
# 4. TRAIN / TEST SPLIT
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─────────────────────────────────────────
# 5. TRAIN BOTH MODELS
# ─────────────────────────────────────────
rf = RandomForestRegressor(
    n_estimators=300, max_depth=None,
    min_samples_split=2, min_samples_leaf=1,
    max_features='sqrt', random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)

gb = GradientBoostingRegressor(
    n_estimators=500, learning_rate=0.05,
    max_depth=6, subsample=0.8,
    min_samples_split=4, min_samples_leaf=2,
    random_state=42
)
gb.fit(X_train, y_train)

# Evaluate
r2_rf  = r2_score(y_test, rf.predict(X_test))
mae_rf = mean_absolute_error(y_test, rf.predict(X_test))
r2_gb  = r2_score(y_test, gb.predict(X_test))
mae_gb = mean_absolute_error(y_test, gb.predict(X_test))

print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)
print(f"  Random Forest     → R²: {r2_rf:.4f} | MAE: {mae_rf:,.2f} EGP")
print(f"  Gradient Boosting → R²: {r2_gb:.4f} | MAE: {mae_gb:,.2f} EGP")

# Pick the best model automatically
best_model= gb if r2_gb >= r2_rf else rf
best_model_name = "Gradient Boosting" if r2_gb >= r2_rf else "Random Forest"
print(f"\n  Using best model: {best_model_name}")

# ─────────────────────────────────────────
# 6. INTERACTIVE USER INPUT
# ─────────────────────────────────────────
KNOWN_BRANDS   = sorted(df['brand'].unique().tolist())
KNOWN_WEBSITES = sorted(df['website'].unique().tolist())

def ask_brand():
    print("\n" + "─" * 60)
    print("  STEP 1 — Brand")
    print("─" * 60)
    print("  Available brands:")
    for i, b in enumerate(KNOWN_BRANDS, 1):
        print(f"    [{i}] {b.title()}")
    while True:
        choice = input("\n  Enter brand name or number: ").strip().lower()
        if choice.isdigit() and 1 <= int(choice) <= len(KNOWN_BRANDS):
            return KNOWN_BRANDS[int(choice) - 1]
        elif choice in KNOWN_BRANDS:
            return choice
        else:
            print("  ⚠️  Not recognised. Please pick from the list above.")

def ask_ram():
    print("\n" + "─" * 60)
    print("  STEP 2 — RAM (GB)")
    print("─" * 60)
    common = [2, 3, 4, 6, 8, 12, 16]
    print("  Common options: " + ", ".join(str(r) for r in common))
    while True:
        val = input("\n  Enter RAM in GB (e.g. 8): ").strip()
        if val.isdigit() and int(val) > 0:
            return int(val)
        print("  ⚠️  Please enter a valid positive number.")

def ask_storage():
    print("\n" + "─" * 60)
    print("  STEP 3 — Storage (GB)")
    print("─" * 60)
    common = [32, 64, 128, 256, 512]
    print("  Common options: " + ", ".join(str(s) for s in common))
    while True:
        val = input("\n  Enter Storage in GB (e.g. 128): ").strip()
        if val.isdigit() and int(val) > 0:
            return int(val)
        print("  ⚠️  Please enter a valid positive number.")

def find_product(brand, ram, storage, website):
    """Find the closest matching product name and URL from the dataset."""
    subset = df[
        (df['brand'] == brand) &
        (df['ram_gb'] == ram) &
        (df['storage_gb'] == storage) &
        (df['website'] == website)
    ].copy()

    if subset.empty:
        # Relax: match brand + website only, pick closest RAM/storage
        subset = df[
            (df['brand'] == brand) &
            (df['website'] == website)
        ].copy()

    if subset.empty:
        return 'N/A', 'N/A'

    # Pick the most recent listing
    subset = subset.sort_values('timestamp', ascending=False)
    row = subset.iloc[0]
    return row['name'].strip(), row['URL']

def predict_for_tablet(NEW_BRAND, NEW_RAM, NEW_STORAGE):
    today = datetime.today()
    year, month, day, dayofweek = today.year, today.month, today.day, today.weekday()

    brand_enc = le_brand.transform([NEW_BRAND])[0]
    websites  = le_website.classes_

    results = []
    for ws in websites:
        ws_enc  = le_website.transform([ws])[0]
        avg_spe = df[df['website'] == ws]['storage_per_egp'].mean()

        input_row = pd.DataFrame([[
            NEW_RAM, NEW_STORAGE,
            brand_enc, ws_enc,
            year, month, day, dayofweek,
            NEW_RAM / (NEW_STORAGE + 1),
            avg_spe
        ]], columns=FEATURES)

        pred = best_model.predict(input_row)[0]
        product_name, product_url = find_product(NEW_BRAND, NEW_RAM, NEW_STORAGE, ws)

        results.append({
            'Website'             : ws.upper(),
            'Product Name'        : product_name,
            'Predicted Price (EGP)': round(pred, 2),
            'URL'                 : product_url
        })

    results_df = pd.DataFrame(results).sort_values('Predicted Price (EGP)').reset_index(drop=True)

    medals = ['🥇', '🥈', '🥉']
    ranks  = [medals[i] if i < 3 else f'  {i+1}.' for i in range(len(results_df))]

    best   = results_df.iloc[0]
    worst  = results_df.iloc[-1]
    saving = worst['Predicted Price (EGP)'] - best['Predicted Price (EGP)']

    print("\n" + "=" * 60)
    print(f"  💰 PREDICTED PRICES — {NEW_BRAND.upper()} | {NEW_RAM}GB RAM | {NEW_STORAGE}GB")
    print("=" * 60)
    for i, (_, r) in enumerate(results_df.iterrows()):
        print(f"\n  {ranks[i]}  {r['Website']}")
        print(f"      📦 Product : {r['Product Name']}")
        print(f"      💵 Price   : EGP {r['Predicted Price (EGP)']:,.2f}")
        print(f"      🔗 URL     : {r['URL']}")

    print("\n" + "─" * 60)
    print(f"  ✅ Best deal     : {best['Website']} → EGP {best['Predicted Price (EGP)']:,.2f}")
    print(f"      🔗 {best['URL']}")
    print(f"  ❌ Most expensive: {worst['Website']} → EGP {worst['Predicted Price (EGP)']:,.2f}")
    print(f"  💸 You can save : EGP {saving:,.2f} by choosing {best['Website']}")
    print("─" * 60)

# ── Main loop ────────────────────────────
print("\n" + "=" * 60)
print("       🔍 TABLET PRICE PREDICTOR")
print("=" * 60)
print("  Answer each question to get predicted prices")
print("  with product names and links across all websites.\n")

while True:
    NEW_BRAND   = ask_brand()
    NEW_RAM     = ask_ram()
    NEW_STORAGE = ask_storage()

    # Summary before predicting
    print("\n" + "=" * 60)
    print("  📋 YOUR TABLET SUMMARY")
    print("=" * 60)
    print(f"  Brand   : {NEW_BRAND.title()}")
    print(f"  RAM     : {NEW_RAM} GB")
    print(f"  Storage : {NEW_STORAGE} GB")
    print("=" * 60)
    input("\n  Press Enter to predict prices...")

    predict_for_tablet(NEW_BRAND, NEW_RAM, NEW_STORAGE)

    # Ask if user wants to predict again
    print()
    again = input("  🔄 Predict another tablet? (y/n): ").strip().lower()
    if again != 'y':
        print("\n  👋 Thanks for using Tablet Price Predictor. Goodbye!")
        break
    print("\n" + "=" * 60)
    print("  🔁 Starting new prediction...")
    print("=" * 60)
