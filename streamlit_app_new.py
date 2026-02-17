import streamlit as st
import pandas as pd
import numpy as np
import warnings
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Tablet Price Tracker",
    page_icon="📊",
    layout="wide"
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: #080c14; color: #e8eaf0; }

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem; font-weight: 800;
    background: linear-gradient(135deg, #00d4ff 0%, #7b5cf0 60%, #ff6b9d 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1.1; margin-bottom: 0.3rem;
}
.hero-sub {
    font-size: 1rem; color: #6b7280; font-weight: 300;
    letter-spacing: 0.04em; margin-bottom: 1.5rem;
}
.stat-card {
    background: #111827; border: 1px solid #1e2535;
    border-radius: 14px; padding: 1rem 1.2rem; text-align: center;
}
.stat-label { font-size: 0.72rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.1em; }
.stat-value { font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 700; }

.signal-card {
    border-radius: 16px; padding: 1.2rem 1.5rem; margin: 0.5rem 0;
    font-family: 'Syne', sans-serif;
}
.signal-buy   { background: linear-gradient(135deg,#0d2b1d,#0b1a12); border: 1.5px solid #00ff8855; }
.signal-wait  { background: linear-gradient(135deg,#2b1a0d,#1a0b0b); border: 1.5px solid #ff6b6b55; }
.signal-neutral { background: linear-gradient(135deg,#1a1a2b,#0d0d1a); border: 1.5px solid #7b5cf055; }

.product-card {
    background: #111827; border: 1px solid #1e2535;
    border-radius: 12px; padding: 0.9rem 1.2rem; margin-bottom: 0.5rem;
    cursor: pointer; transition: border-color 0.2s;
}
.confidence-low    { color: #ff6b6b; }
.confidence-medium { color: #ffd166; }
.confidence-high   { color: #00ff88; }

.stButton > button {
    background: linear-gradient(135deg, #7b5cf0, #00d4ff) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; width: 100% !important;
}
div[data-testid="stSelectbox"] > div {
    background: #111827 !important; border: 1px solid #1e2535 !important;
    border-radius: 10px !important;
}
.stTextInput > div > div > input {
    background: #111827 !important; border: 1px solid #1e2535 !important;
    border-radius: 10px !important; color: #e8eaf0 !important;
}
hr { border-color: #1e2535 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# 1. LOAD & PREPROCESS DATA
# ─────────────────────────────────────────
@st.cache_data(ttl=86400)
def load_data():
    df = pd.read_csv('tablets_cleaned_clean.csv')

    # Clean price
    df['price'] = df['price'].str.replace('EGP', '', regex=False)\
                             .str.replace(',', '', regex=False)\
                             .str.strip().astype(float)

    # Normalize text
    df['brand']   = df['brand'].str.lower().str.strip()
    df['website'] = df['website'].str.lower().str.strip()
    df['name']    = df['name'].str.strip()
    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date']      = df['timestamp'].dt.date

    # Product key: name + website
    df['product_key'] = df['name'].str.lower().str.strip() + ' || ' + df['website']

    # Keep latest URL per product
    df = df.sort_values('timestamp')

    # Average duplicate prices on same day for same product
    df_daily = df.groupby(['product_key', 'date']).agg(
        price     = ('price',   'mean'),
        name      = ('name',    'first'),
        brand     = ('brand',   'first'),
        website   = ('website', 'first'),
        ram_gb    = ('ram_gb',  'first'),
        storage_gb= ('storage_gb','first'),
        URL       = ('URL',     'last'),
        timestamp = ('timestamp','first')
    ).reset_index()

    df_daily['date'] = pd.to_datetime(df_daily['date'])
    df_daily = df_daily.sort_values(['product_key','date'])

    return df_daily


# ─────────────────────────────────────────
# 2. FEATURE ENGINEERING PER PRODUCT
# ─────────────────────────────────────────
def engineer_product(pdf):
    pdf = pdf.sort_values('date').copy()
    pdf['day_index']   = (pdf['date'] - pdf['date'].min()).dt.days
    pdf['dayofweek']   = pdf['date'].dt.dayofweek
    pdf['rolling_avg'] = pdf['price'].rolling(window=3, min_periods=1).mean()
    pdf['pct_change']  = pdf['price'].pct_change().fillna(0)
    pdf['volatility']  = pdf['price'].rolling(window=3, min_periods=1).std().fillna(0)
    return pdf


# ─────────────────────────────────────────
# 3. FORECAST FUNCTION
# ─────────────────────────────────────────
def forecast_product(pdf, days_ahead=7):
    pdf = engineer_product(pdf)
    n   = len(pdf)

    X = pdf['day_index'].values.reshape(-1, 1)
    y = pdf['price'].values

    last_day  = pdf['day_index'].max()
    last_date = pdf['date'].max()
    last_price = pdf['price'].iloc[-1]
    avg_price  = pdf['price'].mean()
    min_price  = pdf['price'].min()
    max_price  = pdf['price'].max()

    # Choose model based on data size
    if n >= 10:
        poly  = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model  = LinearRegression().fit(X_poly, y)
        y_fit  = model.predict(X_poly)
        future_days = np.array([[last_day + i] for i in range(1, days_ahead + 1)])
        future_poly = poly.transform(future_days)
        forecast    = model.predict(future_poly)
        model_type  = "polynomial"
    else:
        model = LinearRegression().fit(X, y)
        y_fit = model.predict(X)
        future_days = np.array([[last_day + i] for i in range(1, days_ahead + 1)])
        forecast    = model.predict(future_days)
        model_type  = "linear"

    # Residuals for confidence band
    residuals = np.abs(y - y_fit)
    mae       = residuals.mean()
    std       = residuals.std() if len(residuals) > 1 else mae

    # Clip forecast to realistic range (not below 50% of min, not above 150% of max)
    forecast = np.clip(forecast, min_price * 0.5, max_price * 1.5)

    # Forecast dates — start from TODAY (the moment the user searches)
    today = pd.Timestamp.today().normalize()
    forecast_dates = [today + timedelta(days=i) for i in range(1, days_ahead + 1)]

    # Trend slope
    slope = (forecast[-1] - last_price) / days_ahead

    # Confidence based on data points
    if n >= 15:
        confidence = "High"
    elif n >= 7:
        confidence = "Medium"
    else:
        confidence = "Low"

    # Buy signal logic
    price_vs_avg = (last_price - avg_price) / avg_price * 100
    trend_pct    = (forecast[-1] - last_price) / last_price * 100

    if price_vs_avg <= -3 and trend_pct >= 0:
        signal      = "buy"
        signal_text = "🟢 Good Time to Buy"
        signal_desc = f"Current price is {abs(price_vs_avg):.1f}% below average and is expected to rise. Buy now before it goes up."
    elif price_vs_avg >= 3 and trend_pct < 0:
        signal      = "wait"
        signal_text = "🔴 Wait — Price May Drop"
        signal_desc = f"Current price is {price_vs_avg:.1f}% above average and the trend shows it may decrease. Consider waiting."
    elif abs(trend_pct) <= 2:
        signal      = "neutral"
        signal_text = "🟡 Price is Stable"
        signal_desc = f"Price is not expected to change significantly in the next {days_ahead} days. Current price is close to the average."
    elif trend_pct > 2:
        signal      = "wait"
        signal_text = "🔴 Price Rising — Buy Soon or Wait"
        signal_desc = f"Price is trending upward (~{trend_pct:.1f}% over {days_ahead} days). Buy soon if you need it, or wait for a potential correction."
    else:
        signal      = "buy"
        signal_text = "🟢 Price is Dropping — Good Time to Buy"
        signal_desc = f"Price is expected to drop ~{abs(trend_pct):.1f}% over the next {days_ahead} days."

    return {
        'pdf'            : pdf,
        'forecast_dates' : forecast_dates,
        'forecast_prices': forecast,
        'mae'            : mae,
        'std'            : std,
        'slope'          : slope,
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
# 4. CHART BUILDERS
# ─────────────────────────────────────────
CHART_BG   = "#080c14"
CHART_GRID = "#1e2535"

def chart_price_history(result):
    pdf   = result['pdf']
    fdates = result['forecast_dates']
    fprices = result['forecast_prices']
    mae   = result['mae']

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=pdf['date'], y=pdf['price'],
        mode='lines+markers',
        name='Historical Price',
        line=dict(color='#00d4ff', width=2.5),
        marker=dict(size=6, color='#00d4ff'),
        hovertemplate='%{x}<br>EGP %{y:,.0f}<extra></extra>'
    ))

    # Rolling average
    fig.add_trace(go.Scatter(
        x=pdf['date'], y=pdf['rolling_avg'],
        mode='lines', name='3-day Avg',
        line=dict(color='#7b5cf0', width=1.5, dash='dot'),
        hovertemplate='%{x}<br>Avg EGP %{y:,.0f}<extra></extra>'
    ))

    last_hist_date  = pdf['date'].iloc[-1]
    last_hist_price = pdf['price'].iloc[-1]
    first_fore_date  = fdates[0]
    first_fore_price = fprices[0]

    fig.add_trace(go.Scatter(
        x=[last_hist_date, first_fore_date],
        y=[last_hist_price, first_fore_price],
        mode='lines',
        name='Bridge',
        line=dict(color='#ffd166', width=1.5, dash='dot'),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=fdates, y=fprices,
        mode='lines+markers', name='Forecast',
        line=dict(color='#ffd166', width=2.5, dash='dash'),
        marker=dict(size=6, symbol='diamond', color='#ffd166'),
        hovertemplate='%{x}<br>Forecast EGP %{y:,.0f}<extra></extra>'
    ))

    band_dates  = [last_hist_date, first_fore_date] + fdates[1:]
    band_prices = [last_hist_price, first_fore_price] + list(fprices[1:])
    upper = [p + mae for p in band_prices]
    lower = [max(p - mae, 0) for p in band_prices]

    fig.add_trace(go.Scatter(
        x=band_dates + band_dates[::-1],
        y=upper + lower[::-1],
        fill='toself',
        fillcolor='rgba(255,209,102,0.08)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Confidence Band',
        hoverinfo='skip'
    ))
    today_str = str(pd.Timestamp.today().date())
    fig.add_shape(
        type="line",
        x0=today_str, x1=today_str,
        y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color="#444", width=1, dash="dot")
    )
    fig.add_annotation(
        x=today_str, y=1,
        xref="x", yref="paper",
        text="Today", showarrow=False,
        font=dict(color="#888", size=11),
        yanchor="bottom"
    )

    fig.update_layout(
        title=dict(text='📈 Price History & 7-Day Forecast', font=dict(family='Syne', size=16, color='#e8eaf0')),
        plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG,
        font=dict(family='DM Sans', color='#8b92a5'),
        xaxis=dict(gridcolor=CHART_GRID, showgrid=True, zeroline=False, title='Date'),
        yaxis=dict(gridcolor=CHART_GRID, showgrid=True, zeroline=False, title='Price (EGP)',
                   tickformat=',.0f'),
        legend=dict(bgcolor='#111827', bordercolor='#1e2535', borderwidth=1),
        hovermode='x unified',
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig
    
def chart_price_distribution(pdf):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=pdf['price'], nbinsx=15,
        marker=dict(color='#7b5cf0', opacity=0.8, line=dict(color='#080c14', width=1)),
        name='Price Distribution',
        hovertemplate='EGP %{x:,.0f}<br>Count: %{y}<extra></extra>'
    ))
    mean_price = pdf['price'].mean()
    curr_price = pdf['price'].iloc[-1]
    fig.add_shape(type="line", x0=mean_price, x1=mean_price, y0=0, y1=1,
                  xref="x", yref="paper", line=dict(color="#00d4ff", width=1.5, dash="dash"))
    fig.add_annotation(x=mean_price, y=1, xref="x", yref="paper",
                       text="Mean", showarrow=False, font=dict(color="#00d4ff", size=11), yanchor="bottom")
    fig.add_shape(type="line", x0=curr_price, x1=curr_price, y0=0, y1=1,
                  xref="x", yref="paper", line=dict(color="#ffd166", width=1.5, dash="dot"))
    fig.add_annotation(x=curr_price, y=0.85, xref="x", yref="paper",
                       text="Current", showarrow=False, font=dict(color="#ffd166", size=11), yanchor="bottom")

    fig.update_layout(
        title=dict(text='📊 Price Distribution', font=dict(family='Syne', size=16, color='#e8eaf0')),
        plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG,
        font=dict(family='DM Sans', color='#8b92a5'),
        xaxis=dict(gridcolor=CHART_GRID, title='Price (EGP)', tickformat=',.0f'),
        yaxis=dict(gridcolor=CHART_GRID, title='Count'),
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig


def chart_price_volatility(pdf):
    colors = ['#ff6b6b' if c > 0 else '#00ff88' for c in pdf['pct_change']]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=pdf['date'], y=pdf['pct_change'] * 100,
        marker_color=colors,
        name='Daily Change %',
        hovertemplate='%{x}<br>%{y:.2f}%<extra></extra>'
    ))
    fig.add_shape(type="line", x0=0, x1=1, y0=0, y1=0,
                  xref="paper", yref="y", line=dict(color="#444", width=1))

    fig.update_layout(
        title=dict(text='📉 Daily Price Change %', font=dict(family='Syne', size=16, color='#e8eaf0')),
        plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG,
        font=dict(family='DM Sans', color='#8b92a5'),
        xaxis=dict(gridcolor=CHART_GRID, title='Date'),
        yaxis=dict(gridcolor=CHART_GRID, title='Change %', ticksuffix='%'),
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig


def chart_all_products_overview(df):
    """Average price per brand over time"""
    brand_time = df.groupby(['brand', 'date'])['price'].mean().reset_index()
    brand_time['date'] = pd.to_datetime(brand_time['date'])

    BRAND_COLORS = ['#00d4ff','#7b5cf0','#ff6b9d','#ffd166','#00ff88',
                    '#ff9966','#a78bfa','#34d399','#f472b6','#60a5fa','#fb923c']
    fig = go.Figure()
    for idx, brand in enumerate(brand_time['brand'].unique()):
        bdf = brand_time[brand_time['brand'] == brand]
        fig.add_trace(go.Scatter(
            x=bdf['date'], y=bdf['price'],
            mode='lines', name=brand.title(),
            line=dict(color=BRAND_COLORS[idx % len(BRAND_COLORS)], width=2),
            hovertemplate=f'{brand.title()}<br>%{{x}}<br>EGP %{{y:,.0f}}<extra></extra>'
        ))

    fig.update_layout(
        title=dict(text='🏷️ Average Price by Brand Over Time', font=dict(family='Syne', size=16, color='#e8eaf0')),
        plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG,
        font=dict(family='DM Sans', color='#8b92a5'),
        xaxis=dict(gridcolor=CHART_GRID, title='Date'),
        yaxis=dict(gridcolor=CHART_GRID, title='Avg Price (EGP)', tickformat=',.0f'),
        legend=dict(bgcolor='#111827', bordercolor='#1e2535', borderwidth=1),
        hovermode='x unified',
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig


def chart_website_price_box(df):
    fig = go.Figure()
    colors = ['#00d4ff','#7b5cf0','#ff6b9d','#ffd166','#00ff88']
    for idx, ws in enumerate(sorted(df['website'].unique())):
        wdf = df[df['website'] == ws]
        fig.add_trace(go.Box(
            y=wdf['price'], name=ws.upper(),
            marker_color=colors[idx % len(colors)],
            boxmean=True,
            hovertemplate=f'{ws.upper()}<br>EGP %{{y:,.0f}}<extra></extra>'
        ))
    fig.update_layout(
        title=dict(text='🛒 Price Range by Website', font=dict(family='Syne', size=16, color='#e8eaf0')),
        plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG,
        font=dict(family='DM Sans', color='#8b92a5'),
        xaxis=dict(gridcolor=CHART_GRID),
        yaxis=dict(gridcolor=CHART_GRID, title='Price (EGP)', tickformat=',.0f'),
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig


# ─────────────────────────────────────────
# 5. LOAD DATA
# ─────────────────────────────────────────
df = load_data()

# ─────────────────────────────────────────
# 6. UI — HEADER
# ─────────────────────────────────────────
st.markdown('<div class="hero-title">📊 Tablet Price Tracker</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Track price history · Forecast next 7 days · Know when to buy</div>', unsafe_allow_html=True)

last_data_date = pd.to_datetime(df['date'].max()).strftime('%B %d, %Y')
st.markdown(f"""
<div style="text-align:center; color:#6b7280; font-size:0.8rem; margin-top:-0.5rem; margin-bottom:1rem;">
    📅 Data last updated: {last_data_date}
</div>
""", unsafe_allow_html=True

# ─────────────────────────────────────────
# 7. TABS
# ─────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Product Forecast", "📊 Data Overview", "📈 Market Insights"])

# ══════════════════════════════════════════
# TAB 1 — PRODUCT FORECAST
# ══════════════════════════════════════════
with tab1:
    st.markdown("### Search for a Tablet")

    col_s1, col_s2, col_s3 = st.columns([2, 1, 1])
    with col_s1:
        search  = st.text_input("🔎 Product name", placeholder="e.g. iPad, Samsung, Huawei...")
    with col_s2:
        brands  = ['All'] + sorted([b.title() for b in df['brand'].unique()])
        brand_f = st.selectbox("Brand", brands)
    with col_s3:
        websites = ['All'] + sorted([w.upper() for w in df['website'].unique()])
        website_f = st.selectbox("Website", websites)

    # Filter products
    filtered = df.copy()
    if search:
        filtered = filtered[filtered['name'].str.lower().str.contains(search.lower(), na=False)]
    if brand_f != 'All':
        filtered = filtered[filtered['brand'] == brand_f.lower()]
    if website_f != 'All':
        filtered = filtered[filtered['website'] == website_f.lower()]

    # Get unique products with observation count
    product_summary = filtered.groupby('product_key').agg(
        name      = ('name', 'first'),
        website   = ('website', 'first'),
        brand     = ('brand', 'first'),
        n_obs     = ('price', 'count'),
        last_price= ('price', 'last'),
        min_price = ('price', 'min'),
        max_price = ('price', 'max'),
    ).reset_index().sort_values('n_obs', ascending=False)

    if product_summary.empty:
        st.warning("No products found. Try a different search.")
    else:
        st.markdown(f"**{len(product_summary)} products found** — select one to see forecast")
        st.markdown("")

        # Show product list
        selected_key = st.selectbox(
            "Select a product",
            options=product_summary['product_key'].tolist(),
            format_func=lambda k: (
                f"{product_summary[product_summary['product_key']==k]['name'].values[0]} "
                f"| {product_summary[product_summary['product_key']==k]['website'].values[0].upper()} "
                f"({product_summary[product_summary['product_key']==k]['n_obs'].values[0]} observations)"
            )
        )

        if selected_key:
            pdf = df[df['product_key'] == selected_key].copy()
            result = forecast_product(pdf, days_ahead=7)

            try:
                result = forecast_product(pdf, days_ahead=7)
            except Exception as e:
                st.error(f"⚠️ Unable to forecast this product. The data may be incomplete or corrupted.")
                st.info(f"Technical details: {str(e)}")
                st.stop()

            st.markdown("---")

            # ── Stats Row ──
            conf_color = {
                'High': '#00ff88', 'Medium': '#ffd166', 'Low': '#ff9966'
            }[result['confidence']]

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.markdown(f"""<div class="stat-card">
                <div class="stat-label">Current Price</div>
                <div class="stat-value" style="color:#00d4ff">EGP {result['last_price']:,.0f}</div>
            </div>""", unsafe_allow_html=True)
            c2.markdown(f"""<div class="stat-card">
                <div class="stat-label">7-Day Forecast</div>
                <div class="stat-value" style="color:#ffd166">EGP {result['forecast_prices'][-1]:,.0f}</div>
            </div>""", unsafe_allow_html=True)
            c3.markdown(f"""<div class="stat-card">
                <div class="stat-label">Avg Price</div>
                <div class="stat-value" style="color:#7b5cf0">EGP {result['avg_price']:,.0f}</div>
            </div>""", unsafe_allow_html=True)
            c4.markdown(f"""<div class="stat-card">
                <div class="stat-label">Min / Max</div>
                <div class="stat-value" style="color:#8b92a5; font-size:1rem">
                    {result['min_price']:,.0f} / {result['max_price']:,.0f}
                </div>
            </div>""", unsafe_allow_html=True)
            c5.markdown(f"""<div class="stat-card">
                <div class="stat-label">Confidence</div>
                <div class="stat-value" style="color:{conf_color}">{result['confidence']}</div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Buy Signal ──
            signal_class = {
                'buy': 'signal-buy', 'wait': 'signal-wait', 'neutral': 'signal-neutral'
            }[result['signal']]

            trend_arrow = "⬆️" if result['trend_pct'] > 1 else ("⬇️" if result['trend_pct'] < -1 else "➡️")
            trend_color = "#ff6b6b" if result['trend_pct'] > 1 else ("#00ff88" if result['trend_pct'] < -1 else "#ffd166")

            st.markdown(f"""
            <div class="signal-card {signal_class}">
                <div style="font-size:1.3rem; font-weight:800; margin-bottom:0.4rem;">
                    {result['signal_text']}
                </div>
                <div style="font-size:0.95rem; color:#c0c4d0; font-family:'DM Sans';">
                    {result['signal_desc']}
                </div>
                <div style="margin-top:0.8rem; font-size:0.85rem; color:#8b92a5; font-family:'DM Sans';">
                    {trend_arrow} Expected change over 7 days:
                    <span style="color:{trend_color}; font-weight:700;">
                        {result['trend_pct']:+.1f}%
                    </span>
                    &nbsp;|&nbsp;
                    Based on <strong>{result['n_obs']}</strong> price observations
                    &nbsp;|&nbsp;
                    Confidence: <span style="color:{conf_color}; font-weight:700;">{result['confidence']}</span>
                    &nbsp;|&nbsp;
                    ±EGP {result['mae']:,.0f} margin
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Tomorrow's Forecast ──
            tomorrow = result['forecast_prices'][0]
            tomorrow_diff = tomorrow - result['last_price']
            diff_color = "#ff6b6b" if tomorrow_diff > 0 else "#00ff88"
            diff_sign  = "+" if tomorrow_diff > 0 else ""

            st.markdown(f"""
            <div style="background:#111827; border:1px solid #1e2535; border-radius:14px;
                        padding:1rem 1.5rem; margin-bottom:1rem; display:flex;
                        justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1rem;">
                <div>
                    <div style="font-size:0.75rem; color:#6b7280; text-transform:uppercase;
                                letter-spacing:0.1em;">Tomorrow's Expected Price</div>
                    <div style="font-family:'Syne',sans-serif; font-size:1.8rem;
                                font-weight:800; color:#ffd166;">
                        EGP {tomorrow:,.0f}
                    </div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:0.75rem; color:#6b7280;">vs Today</div>
                    <div style="font-family:'Syne',sans-serif; font-size:1.3rem;
                                font-weight:700; color:{diff_color};">
                        {diff_sign}EGP {abs(tomorrow_diff):,.0f}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Charts ──
            st.plotly_chart(chart_price_history(result), use_container_width=True)

            col_ch1, col_ch2 = st.columns(2)
            with col_ch1:
                st.plotly_chart(chart_price_distribution(result['pdf']), use_container_width=True)
            with col_ch2:
                st.plotly_chart(chart_price_volatility(result['pdf']), use_container_width=True)

            # ── 7-Day Forecast Table ──
            st.markdown("#### 📅 7-Day Forecast Breakdown")
            forecast_df = pd.DataFrame({
                'Date'           : result['forecast_dates'],
                'Expected Price' : [f"EGP {p:,.0f}" for p in result['forecast_prices']],
                'Low Estimate'   : [f"EGP {max(p - result['mae'], 0):,.0f}" for p in result['forecast_prices']],
                'High Estimate'  : [f"EGP {p + result['mae']:,.0f}" for p in result['forecast_prices']],
            })
            forecast_df['Date'] = forecast_df['Date'].apply(lambda d: d.strftime('%a, %b %d'))
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)

            # ── Product Link ──
            url = pdf['URL'].iloc[-1]
            if url and str(url) != 'nan':
                st.markdown(f"""
                <div style="margin-top:1rem; padding:0.8rem 1.2rem; background:#111827;
                            border:1px solid #1e2535; border-radius:10px;">
                    <span style="color:#6b7280; font-size:0.85rem;">🔗 Product Link: </span>
                    <a href="{url}" target="_blank"
                       style="color:#7b5cf0; font-size:0.85rem; word-break:break-all;">
                        {url}
                    </a>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════
# TAB 2 — DATA OVERVIEW
# ══════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Dataset Overview")

    # Top stats
    d1, d2, d3, d4 = st.columns(4)
    d1.markdown(f"""<div class="stat-card">
        <div class="stat-label">Total Records</div>
        <div class="stat-value" style="color:#00d4ff">{len(df):,}</div>
    </div>""", unsafe_allow_html=True)
    d2.markdown(f"""<div class="stat-card">
        <div class="stat-label">Unique Products</div>
        <div class="stat-value" style="color:#7b5cf0">{df['product_key'].nunique()}</div>
    </div>""", unsafe_allow_html=True)
    d3.markdown(f"""<div class="stat-card">
        <div class="stat-label">Date Range</div>
        <div class="stat-value" style="color:#ffd166; font-size:0.9rem">
            {pd.to_datetime(df['date'].min()).strftime('%b %d')} → {pd.to_datetime(df['date'].max()).strftime('%b %d, %Y')}
        </div>
    </div>""", unsafe_allow_html=True)
    d4.markdown(f"""<div class="stat-card">
        <div class="stat-label">Websites Tracked</div>
        <div class="stat-value" style="color:#ff6b9d">{df['website'].nunique()}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts
    st.plotly_chart(chart_all_products_overview(df), use_container_width=True)
    st.plotly_chart(chart_website_price_box(df), use_container_width=True)

    # Price observations per product histogram
    obs_counts = df.groupby('product_key')['price'].count().reset_index()
    obs_counts.columns = ['product_key', 'observations']
    fig_obs = go.Figure(go.Histogram(
        x=obs_counts['observations'],
        nbinsx=20,
        marker=dict(color='#7b5cf0', opacity=0.85, line=dict(color='#080c14', width=1)),
        hovertemplate='Observations: %{x}<br>Products: %{y}<extra></extra>'
    ))
    fig_obs.update_layout(
        title=dict(text='📦 How Many Price Points Per Product',
                   font=dict(family='Syne', size=16, color='#e8eaf0')),
        plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG,
        font=dict(family='DM Sans', color='#8b92a5'),
        xaxis=dict(gridcolor=CHART_GRID, title='Number of Observations'),
        yaxis=dict(gridcolor=CHART_GRID, title='Number of Products'),
        margin=dict(l=10, r=10, t=50, b=10)
    )
    st.plotly_chart(fig_obs, use_container_width=True)


# ══════════════════════════════════════════
# TAB 3 — MARKET INSIGHTS
# ══════════════════════════════════════════
with tab3:
    st.markdown("### 📈 Market Insights")
    st.markdown("Which products had the biggest price changes over the tracked period?")

    # Price change per product
    changes = []
    for key, grp in df.groupby('product_key'):
        grp = grp.sort_values('date')
        if len(grp) >= 3:
            first = grp['price'].iloc[0]
            last  = grp['price'].iloc[-1]
            if first > 0:
                pct = (last - first) / first * 100
            else:
                continue
            changes.append({
                'Product'      : grp['name'].iloc[0],
                'Website'      : grp['website'].iloc[0].upper(),
                'Start Price'  : first,
                'Current Price': last,
                'Change %'     : round(pct, 1),
                'Observations' : len(grp)
            })

    changes_df = pd.DataFrame(changes).sort_values('Change %').reset_index(drop=True)

    if not changes_df.empty:
        col_i1, col_i2 = st.columns(2)

        with col_i1:
            st.markdown("#### 🟢 Biggest Price Drops")
            drops = changes_df.head(5)[['Product','Website','Change %','Current Price']].copy()
            drops['Current Price'] = drops['Current Price'].apply(lambda x: f"EGP {x:,.0f}")
            drops['Change %']      = drops['Change %'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(drops, use_container_width=True, hide_index=True)

        with col_i2:
            st.markdown("#### 🔴 Biggest Price Rises")
            rises = changes_df.tail(5).iloc[::-1][['Product','Website','Change %','Current Price']].copy()
            rises['Current Price'] = rises['Current Price'].apply(lambda x: f"EGP {x:,.0f}")
            rises['Change %']      = rises['Change %'].apply(lambda x: f"+{x:.1f}%")
            st.dataframe(rises, use_container_width=True, hide_index=True)

        # Bar chart — build colors list separately to avoid plotly issues
        top_changes = pd.concat([
            changes_df.head(10), changes_df.tail(10)
        ]).drop_duplicates().reset_index(drop=True)

        top_changes['label'] = (
            top_changes['Product'].str[:28] + ' | ' + top_changes['Website']
        )
        bar_colors = ['#00ff88' if x < 0 else '#ff6b6b'
                      for x in top_changes['Change %'].tolist()]

        fig_changes = go.Figure()
        fig_changes.add_trace(go.Bar(
            x=top_changes['Change %'].tolist(),
            y=top_changes['label'].tolist(),
            orientation='h',
            marker=dict(color=bar_colors),
            hovertemplate='%{y}<br>Change: %{x:.1f}%<extra></extra>'
        ))
        fig_changes.add_shape(
            type="line", x0=0, x1=0, y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color="#666", width=1.5)
        )
        fig_changes.update_layout(
            title=dict(
                text='💹 Price Change % Since First Observation',
                font=dict(family='Syne', size=16, color='#e8eaf0')
            ),
            plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG,
            font=dict(family='DM Sans', color='#8b92a5'),
            xaxis=dict(gridcolor=CHART_GRID, title='Change %', ticksuffix='%'),
            yaxis=dict(gridcolor=CHART_GRID, tickfont=dict(size=10)),
            height=520,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        st.plotly_chart(fig_changes, use_container_width=True)
    else:
        st.info("Not enough data to calculate price changes yet.")
