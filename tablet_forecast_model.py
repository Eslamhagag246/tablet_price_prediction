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
df = pd.read_csv('tablets_cleaned_clean.csv')
print("=" * 60)
print("ORIGINAL SHAPE:", df.shape)

# ─────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────

# --- 2.1 Clean price
df['price'] = df['price'].str.replace('EGP', '', regex=False)\
                         .str.replace(',', '', regex=False)\
                         .str.strip().astype(float)

# --- 2.2 Normalize text
df['brand']   = df['brand'].str.lower().str.strip()
df['website'] = df['website'].str.lower().str.strip()
df['name']    = df['name'].str.strip()

# --- 2.3 Parse timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['date']      = df['timestamp'].dt.date

# --- 2.4 Create product key (name + website)
# This is the core identity of each product listing
df['product_key'] = df['name'].str.lower().str.strip() + ' || ' + df['website']

# --- 2.5 Average duplicate prices on same day for same product
# Multiple scrapes on the same day → take the mean price
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

# --- 2.6 Observation counts per product
obs_counts = df_daily.groupby('product_key').size()
print(f"\nObservations per product:")
print(f"  Min    : {obs_counts.min()}")
print(f"  Median : {obs_counts.median()}")
print(f"  Max    : {obs_counts.max()}")
print(f"  5+ obs : {(obs_counts >= 5).sum()} products")
print(f"  10+ obs: {(obs_counts >= 10).sum()} products")

# ─────────────────────────────────────────
# 3. FEATURE ENGINEERING (per product)
# ─────────────────────────────────────────
def engineer_features(pdf):
    """
    Build time-based features for a single product's price history.
    pdf: DataFrame with columns [date, price] for one product
    """
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
def forecast_product(pdf, days_ahead=7):
    """
    Fit a trend model on a single product's price history
    and forecast the next `days_ahead` days.

    Model choice:
      - 10+ observations → Polynomial Regression (degree 2) — catches curves
      - <10 observations → Linear Regression — simple, avoids overfitting

    Returns a dict with forecast prices, confidence, buy signal, and metrics.
    """
    pdf = engineer_features(pdf)
    n   = len(pdf)

    X          = pdf['day_index'].values.reshape(-1, 1)
    y          = pdf['price'].values
    last_day   = pdf['day_index'].max()
    last_date  = pdf['date'].max()
    last_price = pdf['price'].iloc[-1]
    avg_price  = pdf['price'].mean()
    min_price  = pdf['price'].min()
    max_price  = pdf['price'].max()

    # ── Choose model based on data size ──
    if n >= 10:
        poly   = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model  = LinearRegression().fit(X_poly, y)
        y_fit  = model.predict(X_poly)

        future_X    = np.array([[last_day + i] for i in range(1, days_ahead + 1)])
        future_poly = poly.transform(future_X)
        forecast    = model.predict(future_poly)
        model_type  = "Polynomial (degree 2)"

    else:
        model    = LinearRegression().fit(X, y)
        y_fit    = model.predict(X)
        future_X = np.array([[last_day + i] for i in range(1, days_ahead + 1)])
        forecast = model.predict(future_X)
        model_type = "Linear"

    # ── Evaluation on training data ──
    mae = mean_absolute_error(y, y_fit)
    r2  = r2_score(y, y_fit)

    # ── Clip forecast to realistic bounds ──
    # Never predict below 50% of min or above 150% of max seen
    forecast = np.clip(forecast, min_price * 0.5, max_price * 1.5)

    today = pd.Timestamp.today().normalize()
    forecast_dates = [today + timedelta(days=i) for i in range(1, days_ahead + 1)]

    # ── Trend: % change from today to day 7 ──
    trend_pct = (forecast[-1] - last_price) / last_price * 100

    # ── Confidence based on number of observations ──
    if n >= 15:
        confidence = "High"
    elif n >= 7:
        confidence = "Medium"
    else:
        confidence = "Low"

    # ── Buy Signal Logic ──
    price_vs_avg = (last_price - avg_price) / avg_price * 100

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
                       f"in the next {days_ahead} days. Current price is fair.")

    elif trend_pct > 2:
        signal      = "wait"
        signal_text = "🔴 Price Rising — Buy Soon or Wait"
        signal_desc = (f"Price is trending upward (~{trend_pct:.1f}% over {days_ahead} days). "
                       f"Buy soon if you need it.")
    else:
        signal      = "buy"
        signal_text = "🟢 Price Dropping — Good Time to Buy"
        signal_desc = (f"Price is expected to drop ~{abs(trend_pct):.1f}% "
                       f"over the next {days_ahead} days.")

    return {
        'pdf'            : pdf,
        'forecast_dates' : forecast_dates,
        'forecast_prices': forecast,
        'mae'            : mae,
        'r2'             : r2,
        'confidence'     : confidence,
        'signal'         : signal,
        'signal_text'    : signal_text,
        'signal_desc'    : signal_desc,
        'last_price'     : last_price,
        'avg_price'      : avg_price,
        'min_price'      : min_price,
        'max_price'      : max_price,
        'trend_pct'      : trend_pct,
        'n_obs'          : n,
        'model_type'     : model_type,
        'price_vs_avg'   : price_vs_avg,
    }

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
