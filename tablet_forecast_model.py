import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import timedelta

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
df = pd.read_csv('tablets_full_continuous_series.csv')
print("=" * 60)
print("ORIGINAL SHAPE:", df.shape)

# ─────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────

df['price'] = df['price'].str.replace('EGP', '', regex=False)\
                         .str.replace(',', '', regex=False)\
                         .str.strip().astype(float)

df['brand']   = df['brand'].str.lower().str.strip()
df['website'] = df['website'].str.lower().str.strip()
df['name']    = df['name'].str.strip()

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['date']      = df['timestamp'].dt.date

df['product_key'] = (
    df['name'].str.lower() + ' ' +
    df['website'].astype(str) + ' ' +
    df['ram_gb'].astype(str) + ' ' +
    df['storage_gb'].astype(str)
)

df_daily = df.groupby(['product_key', 'date']).agg(
    price      = ('price',      'mean'),
    name       = ('name',       'first'),
    brand      = ('brand',      'first'),
    website    = ('website',    'first'),
    ram_gb     = ('ram_gb',     'first'),
    storage_gb = ('storage_gb', 'first'),
    URL        = ('URL',        'last'),
    timestamp  = ('timestamp',  'first')
).reset_index()

df_daily['date'] = pd.to_datetime(df_daily['date'])
df_daily = df_daily.sort_values(['product_key', 'date']).reset_index(drop=True)

print(f"\nAfter daily averaging:")
print(f"  Rows         : {len(df_daily)}")
print(f"  Unique products: {df_daily['product_key'].nunique()}")
print(f"  Date range   : {df_daily['date'].min().date()} → {df_daily['date'].max().date()}")

obs_counts = df_daily.groupby('product_key').size()
print(f"\nObservations per product:")
print(f"  Min    : {obs_counts.min()}")
print(f"  Median : {obs_counts.median()}")
print(f"  Max    : {obs_counts.max()}")
print(f"  5+ obs : {(obs_counts >= 5).sum()} products")
print(f"  10+ obs: {(obs_counts >= 10).sum()} products")
def engineer_features(pdf):
    
    pdf = pdf.sort_values('date').copy()

    # Core time feature: integer day index (0, 1, 2, ...)
    pdf['day_index'] = (pdf['date'] - pdf['date'].min()).dt.days

    # Day of week (0=Monday, 6=Sunday) — captures weekend patterns
    pdf['dayofweek'] = pdf['date'].dt.dayofweek

    # Rolling 3-day average — smooths noise
    pdf['rolling_avg'] = pdf['price'].rolling(window=3, min_periods=1).mean()

    # Daily % change — measures volatility
    pdf['pct_change'] = pdf['price'].pct_change().fillna(0)

    # Rolling volatility (std of last 3 prices)
    pdf['volatility'] = pdf['price'].rolling(window=3, min_periods=1).std().fillna(0)

    # Price relative to product's own average — is it currently cheap or expensive?
    pdf['price_vs_avg'] = (pdf['price'] - pdf['price'].mean()) / pdf['price'].mean() * 100
    return pdf
# ─────────────────────────────────────────
# 4. MODEL: FORECAST FUNCTION (per product)
# ─────────────────────────────────────────
def forecast_product(pdf, days_ahead=7, test_horizon=None):
    
    pdf = engineer_features(pdf)
    n = len(pdf)

    X = pdf['day_index'].values.reshape(-1, 1)
    y = pdf['price'].values
    dates = pdf['date'].values

    # --- Split into train/test (80/20) ---
    split = int(n * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    dates_test = dates[split:]   # actual dates of test set

    # --- Train model on training data only ---
    if len(X_train) >= 10:
        poly = PolynomialFeatures(degree=2)
        X_train_poly = poly.fit_transform(X_train)
        model = LinearRegression().fit(X_train_poly, y_train)
        model_type = "Polynomial (degree 2)"
        # For later prediction on test set:
        if len(X_test) > 0:
            X_test_poly = poly.transform(X_test)
            y_pred_test = model.predict(X_test_poly)
        else:
            y_pred_test = np.array([])
    else:
        model = LinearRegression().fit(X_train, y_train)
        model_type = "Linear"
        if len(X_test) > 0:
            y_pred_test = model.predict(X_test)
        else:
            y_pred_test = np.array([])

    # --- Decide what to return: test set predictions OR future forecast ---
    if test_horizon is not None and len(y_pred_test) > 0:
        # Use test set predictions
        n_test = len(y_test)
        horizon = min(test_horizon, n_test)          # take last 'test_horizon' days
        forecast_prices = y_pred_test[-horizon:]     # predicted values
        forecast_dates  = [pd.to_datetime(d) for d in dates_test[-horizon:]]
        actual_prices   = y_test[-horizon:]          # actual values for comparison
        # Also store the full test predictions for MAE/R² (already computed earlier)
    else:
        # Original future forecast
        last_day = pdf['day_index'].max()
        future_X = np.array([[last_day + i] for i in range(1, days_ahead + 1)])
        if len(X_train) >= 10:
            future_poly = poly.transform(future_X)
            forecast_prices = model.predict(future_poly)
        else:
            forecast_prices = model.predict(future_X)
        forecast_dates = [pdf['date'].max() + timedelta(days=i) for i in range(1, days_ahead + 1)]
        actual_prices = None   # not applicable for future

    # --- Evaluation metrics on full test set ---
    if len(y_test) > 0:
        mae = mean_absolute_error(y_test, y_pred_test)
        r2  = r2_score(y_test, y_pred_test)
    else:
        mae, r2 = np.nan, np.nan

    # --- Clip forecasts to reasonable bounds ---
    min_price = pdf['price'].min()
    max_price = pdf['price'].max()
    forecast_prices = np.clip(forecast_prices, min_price * 0.5, max_price * 1.5)

    # --- Other statistics (using full dataset for context) ---
    last_price = pdf['price'].iloc[-1]
    avg_price  = pdf['price'].mean()
    trend_pct  = ((forecast_prices[-1] - last_price) / last_price * 100) if len(forecast_prices) > 0 else 0

    # Confidence based on total observations
    if n >= 15:
        confidence = "High"
    elif n >= 7:
        confidence = "Medium"
    else:
        confidence = "Low"

    price_vs_avg = (last_price - avg_price) / avg_price * 100

    # --- Buy/Signal logic (same as before, but uses trend from forecast) ---
    if price_vs_avg <= -3 and trend_pct >= 0:
        signal      = "buy"
        signal_text = "🟢 Good Time to Buy"
        signal_desc = (f"Current price is {abs(price_vs_avg):.1f}% below average "
                       f"and is expected to rise. Buy now.")
    elif price_vs_avg >= 3 and trend_pct < 0:
        signal      = "wait"
        signal_text = "🔴 Wait — Price May Drop"
        signal_desc = (f"Current price is {price_vs_avg:.1f}% above average "
                       f"and the trend shows it may decrease.")
    elif abs(trend_pct) <= 2:
        signal      = "neutral"
        signal_text = "🟡 Price is Stable"
        signal_desc = (f"Price is not expected to change significantly "
                       f"in the next {len(forecast_prices)} days. Current price is fair.")
    elif trend_pct > 2:
        signal      = "wait"
        signal_text = "🔴 Price Rising — Buy Soon or Wait"
        signal_desc = (f"Price is trending upward (~{trend_pct:.1f}%). "
                       f"Buy soon if you need it.")
    else:
        signal      = "buy"
        signal_text = "🟢 Price Dropping — Good Time to Buy"
        signal_desc = (f"Price is expected to drop ~{abs(trend_pct):.1f}%.")

    return {
        'pdf'             : pdf,
        'forecast_dates'  : forecast_dates,
        'forecast_prices' : forecast_prices,
        'actual_prices'   : actual_prices,          # new field
        'mae'             : mae,
        'r2'              : r2,
        'confidence'      : confidence,
        'signal'          : signal,
        'signal_text'     : signal_text,
        'signal_desc'     : signal_desc,
        'last_price'      : last_price,
        'avg_price'       : avg_price,
        'min_price'       : min_price,
        'max_price'       : max_price,
        'trend_pct'       : trend_pct,
        'n_obs'           : n,
        'model_type'      : model_type,
        'price_vs_avg'    : price_vs_avg,
    }
all_y_test = []
all_y_pred = []

for key, grp in df_daily.groupby('product_key'):
    
    grp = engineer_features(grp)
    n = len(grp)

    if n < 5:
        continue

    X = grp['day_index'].values.reshape(-1,1)
    y = grp['price'].values

    split = int(n * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if len(X_train) >= 10:
        poly = PolynomialFeatures(degree=2)
        X_train = poly.fit_transform(X_train)
        X_test  = poly.transform(X_test)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    all_y_test.extend(y_test)
    all_y_pred.extend(y_pred)

# ─────────────────────────────────────────
# 5. EVALUATE ON ALL PRODUCTS
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL EVALUATION — ALL PRODUCTS")
print("=" * 60)

results_summary = []
for key, grp in df_daily.groupby('product_key'):
    if len(grp) < 3:
        continue
    res = forecast_product(grp)
    results_summary.append({
        'product'    : grp['name'].iloc[0],
        'website'    : grp['website'].iloc[0],
        'n_obs'      : res['n_obs'],
        'model'      : res['model_type'],
        'mae'        : round(res['mae'], 2),
        'r2'         : round(res['r2'], 4),
        'confidence' : res['confidence'],
        'signal'     : res['signal_text'],
        'trend_pct'  : round(res['trend_pct'], 2),
    })

summary_df = pd.DataFrame(results_summary)

print(f"\nProducts evaluated: {len(summary_df)}")
print(f"\nAverage MAE : EGP {summary_df['mae'].mean():,.2f}")
print(f"Average R²  : {summary_df['r2'].mean():.4f}")
print(f"\nModel types used:")
print(summary_df['model'].value_counts().to_string())
print(f"\nConfidence distribution:")
print(summary_df['confidence'].value_counts().to_string())
print(f"\nBuy signal distribution:")
print(summary_df['signal'].value_counts().to_string())

# ─────────────────────────────────────────
# 6. EXAMPLE — BEST PRODUCT FORECAST
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("EXAMPLE FORECAST — MOST OBSERVED PRODUCT")
print("=" * 60)

best_key = df_daily.groupby('product_key').size().idxmax()
best_pdf  = df_daily[df_daily['product_key'] == best_key]
res       = forecast_product(best_pdf)

print(f"\nProduct    : {best_pdf['name'].iloc[0]}")
print(f"Website    : {best_pdf['website'].iloc[0].upper()}")
print(f"Model      : {res['model_type']}")
print(f"Observations: {res['n_obs']}")
print(f"Confidence : {res['confidence']}")
print(f"MAE        : EGP {res['mae']:,.2f}")
print(f"R²         : {res['r2']:.4f}")
print(f"\nCurrent Price : EGP {res['last_price']:,.2f}")
print(f"Average Price : EGP {res['avg_price']:,.2f}")
print(f"Min / Max     : EGP {res['min_price']:,.2f} / EGP {res['max_price']:,.2f}")
print(f"\n{res['signal_text']}")
print(f"→ {res['signal_desc']}")
print(f"\n7-Day Forecast:")
print("-" * 40)
for date, price in zip(res['forecast_dates'], res['forecast_prices']):
    low  = max(price - res['mae'], 0)
    high = price + res['mae']
    print(f"  {date.strftime('%a %b %d')}  →  EGP {price:>8,.0f}  "
          f"(range: {low:,.0f} – {high:,.0f})")
print("-" * 40)
print(f"  Expected 7-day change: {res['trend_pct']:+.1f}%")

