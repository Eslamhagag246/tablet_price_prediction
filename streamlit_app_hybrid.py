import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime, timedelta
import plotly.graph_objects as go

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Tablet Price Tracker",
    page_icon="📱",
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
    font-family: 'Syne', sans-serif; font-size: 2.6rem; font-weight: 800;
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

.signal-buy   { background: linear-gradient(135deg,#0d2b1d,#0b1a12); border: 1.5px solid #00ff8855; }
.signal-wait  { background: linear-gradient(135deg,#2b1a0d,#1a0b0b); border: 1.5px solid #ff6b6b55; }
.signal-neutral { background: linear-gradient(135deg,#1a1a2b,#0d0d1a); border: 1.5px solid #7b5cf055; }

.model-badge {
    display: inline-block; padding: 3px 10px; border-radius: 12px;
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.05em;
    text-transform: uppercase; margin-left: 8px;
}
.badge-prophet { background: linear-gradient(135deg, #7b5cf0, #00d4ff); color: white; }
.badge-linear { background: #333; color: #aaa; border: 1px solid #555; }

hr { border-color: #1e2535 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
@st.cache_data(ttl=86400)
def load_data():
    df = pd.read_csv('tablets_cleaned_augmented.csv')
    
    df['price'] = df['price'].str.replace('EGP','',regex=False)\
                             .str.replace(',','',regex=False)\
                             .str.strip().astype(float)
    df['brand'] = df['brand'].str.lower().str.strip()
    df['website'] = df['website'].str.lower().str.strip()
    df['name'] = df['name'].str.strip()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['product_key'] = df['name'].str.lower().str.strip() + ' || ' + df['website']
    
    df_daily = df.groupby(['product_key', 'date']).agg(
        price=('price','mean'), name=('name','first'),
        brand=('brand','first'), website=('website','first'),
        ram_gb=('ram_gb','first'), storage_gb=('storage_gb','first'),
        URL=('URL','last')
    ).reset_index()
    
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    df_daily = df_daily.sort_values(['product_key','date'])
    
    return df_daily


# ─────────────────────────────────────────
# 2. PROPHET MODEL
# ─────────────────────────────────────────
def forecast_prophet(pdf, days_ahead=7):
    """Use Prophet for products with 15+ observations"""
    pdf = pdf.sort_values('date').copy()
    
    # Prepare for Prophet
    prophet_df = pdf[['date', 'price']].rename(columns={'date': 'ds', 'price': 'y'})
    
    # Train Prophet
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=0.05,
        interval_width=0.95
    )
    model.fit(prophet_df, verbose=False)
    
    # Forecast
    today = pd.Timestamp.today().normalize()
    future_dates = pd.date_range(start=today + timedelta(days=1), periods=days_ahead, freq='D')
    future_df = pd.DataFrame({'ds': future_dates})
    forecast = model.predict(future_df)
    
    forecast_prices = forecast['yhat'].values
    forecast_lower = forecast['yhat_lower'].values
    forecast_upper = forecast['yhat_upper'].values
    
    # Clip to reasonable bounds
    min_price = pdf['price'].min()
    max_price = pdf['price'].max()
    forecast_prices = np.clip(forecast_prices, min_price * 0.5, max_price * 1.5)
    
    # Calculate MAE from in-sample fit
    in_sample_pred = model.predict(prophet_df)
    mae = np.abs(pdf['price'].values - in_sample_pred['yhat'].values).mean()
    
    return {
        'model_type': 'Prophet',
        'forecast_dates': future_dates,
        'forecast_prices': forecast_prices,
        'forecast_lower': forecast_lower,
        'forecast_upper': forecast_upper,
        'mae': mae,
        'confidence': 'High' if len(pdf) >= 25 else 'Medium',
        'pdf': pdf
    }


# ─────────────────────────────────────────
# 3. LINEAR/POLY MODEL (FALLBACK)
# ─────────────────────────────────────────
def forecast_linear_poly(pdf, days_ahead=7):
    """Use Linear/Poly for products with <15 observations"""
    pdf = pdf.sort_values('date').copy()
    pdf['day_index'] = (pdf['date'] - pdf['date'].min()).dt.days
    
    X = pdf['day_index'].values.reshape(-1, 1)
    y = pdf['price'].values
    
    n = len(pdf)
    
    # Choose model
    if n >= 10:
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)
        y_fit = model.predict(X_poly)
        model_type = 'Polynomial'
    else:
        model = LinearRegression().fit(X, y)
        y_fit = model.predict(X)
        poly = None
        model_type = 'Linear'
    
    # Forecast
    today = pd.Timestamp.today().normalize()
    last_day = pdf['day_index'].max()
    future_indices = np.array([[last_day + i] for i in range(1, days_ahead + 1)])
    
    if poly:
        future_poly = poly.transform(future_indices)
        forecast_prices = model.predict(future_poly)
    else:
        forecast_prices = model.predict(future_indices)
    
    # Clip to reasonable bounds
    min_price = pdf['price'].min()
    max_price = pdf['price'].max()
    forecast_prices = np.clip(forecast_prices, min_price * 0.5, max_price * 1.5)
    
    # MAE
    mae = np.abs(y - y_fit).mean()
    
    # Forecast dates
    future_dates = pd.date_range(start=today + timedelta(days=1), periods=days_ahead, freq='D')
    
    # Confidence based on data
    if n >= 15:
        confidence = 'High'
    elif n >= 7:
        confidence = 'Medium'
    else:
        confidence = 'Low'
    
    return {
        'model_type': model_type,
        'forecast_dates': future_dates,
        'forecast_prices': forecast_prices,
        'forecast_lower': forecast_prices - mae,
        'forecast_upper': forecast_prices + mae,
        'mae': mae,
        'confidence': confidence,
        'pdf': pdf
    }


# ─────────────────────────────────────────
# 4. HYBRID FORECAST SELECTOR
# ─────────────────────────────────────────
def forecast_product(pdf, days_ahead=7):
    """
    Hybrid approach:
    - Use Prophet for products with 15+ observations
    - Use Linear/Poly for products with <15 observations
    """
    n = len(pdf)
    
    try:
        if n >= 15:
            # Use Prophet (advanced model)
            result = forecast_prophet(pdf, days_ahead)
        else:
            # Use Linear/Poly (simple fallback)
            result = forecast_linear_poly(pdf, days_ahead)
    except Exception as e:
        # If Prophet fails, fallback to Linear/Poly
        st.warning(f"Prophet failed, using Linear/Poly fallback. Error: {str(e)[:50]}")
        result = forecast_linear_poly(pdf, days_ahead)
    
    # Add common metrics
    last_price = pdf['price'].iloc[-1]
    avg_price = pdf['price'].mean()
    min_price = pdf['price'].min()
    max_price = pdf['price'].max()
    trend_pct = (result['forecast_prices'][-1] - last_price) / last_price * 100
    price_vs_avg = (last_price - avg_price) / avg_price * 100
    
    # Buy signal
    if price_vs_avg <= -3 and trend_pct >= 0:
        signal = "buy"
        signal_text = "🟢 Good Time to Buy"
        signal_desc = f"Current price is {abs(price_vs_avg):.1f}% below average and expected to rise."
    elif price_vs_avg >= 3 and trend_pct < 0:
        signal = "wait"
        signal_text = "🔴 Wait — Price May Drop"
        signal_desc = f"Current price is {price_vs_avg:.1f}% above average and may decrease."
    elif abs(trend_pct) <= 2:
        signal = "neutral"
        signal_text = "🟡 Price is Stable"
        signal_desc = f"Price not expected to change significantly."
    elif trend_pct > 2:
        signal = "wait"
        signal_text = "🔴 Price Rising"
        signal_desc = f"Price trending up ~{trend_pct:.1f}% over next {days_ahead} days."
    else:
        signal = "buy"
        signal_text = "🟢 Price Dropping"
        signal_desc = f"Price expected to drop ~{abs(trend_pct):.1f}%."
    
    result.update({
        'last_price': last_price,
        'avg_price': avg_price,
        'min_price': min_price,
        'max_price': max_price,
        'trend_pct': trend_pct,
        'price_vs_avg': price_vs_avg,
        'signal': signal,
        'signal_text': signal_text,
        'signal_desc': signal_desc,
        'n_obs': n
    })
    
    return result


# ─────────────────────────────────────────
# 5. CHART
# ─────────────────────────────────────────
def chart_price_history(result):
    pdf = result['pdf']
    fdates = result['forecast_dates']
    fprices = result['forecast_prices']
    flower = result['forecast_lower']
    fupper = result['forecast_upper']
    
    fig = go.Figure()
    
    # Historical
    fig.add_trace(go.Scatter(
        x=pdf['date'], y=pdf['price'],
        mode='lines+markers', name='Historical',
        line=dict(color='#00d4ff', width=2.5),
        marker=dict(size=6, color='#00d4ff')
    ))
    
    # Bridge line
    last_hist_date = pdf['date'].iloc[-1]
    last_hist_price = pdf['price'].iloc[-1]
    first_fore_date = fdates[0]
    first_fore_price = fprices[0]
    
    fig.add_trace(go.Scatter(
        x=[last_hist_date, first_fore_date],
        y=[last_hist_price, first_fore_price],
        mode='lines', showlegend=False,
        line=dict(color='#ffd166', width=1.5, dash='dot'),
        hoverinfo='skip'
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=fdates, y=fprices,
        mode='lines+markers', name='Forecast',
        line=dict(color='#ffd166', width=2.5, dash='dash'),
        marker=dict(size=6, symbol='diamond', color='#ffd166')
    ))
    
    # Confidence band
    band_dates = [last_hist_date, first_fore_date] + list(fdates[1:])
    band_prices_lower = [last_hist_price, flower[0]] + list(flower[1:])
    band_prices_upper = [last_hist_price, fupper[0]] + list(fupper[1:])
    
    fig.add_trace(go.Scatter(
        x=band_dates + band_dates[::-1],
        y=band_prices_upper + band_prices_lower[::-1],
        fill='toself', fillcolor='rgba(255,209,102,0.08)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Confidence Band', hoverinfo='skip'
    ))
    
    # Today marker
    today_str = str(pd.Timestamp.today().date())
    fig.add_shape(
        type="line", x0=today_str, x1=today_str, y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color="#444", width=1, dash="dot")
    )
    
    fig.update_layout(
        title=dict(text='📈 Price History & 7-Day Forecast',
                   font=dict(family='Syne', size=16, color='#e8eaf0')),
        plot_bgcolor='#080c14', paper_bgcolor='#080c14',
        font=dict(family='DM Sans', color='#8b92a5'),
        xaxis=dict(gridcolor='#1e2535', title='Date'),
        yaxis=dict(gridcolor='#1e2535', title='Price (EGP)', tickformat=',.0f'),
        legend=dict(bgcolor='#111827', bordercolor='#1e2535', borderwidth=1),
        hovermode='x unified',
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    return fig


# ─────────────────────────────────────────
# 6. LOAD DATA
# ─────────────────────────────────────────
df = load_data()
KNOWN_BRANDS = sorted(df['brand'].unique().tolist())

# ─────────────────────────────────────────
# 7. UI
# ─────────────────────────────────────────
st.markdown('<div class="hero-title">📱 Tablet Price Tracker</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">AI-powered forecasting · Prophet + Linear/Poly hybrid model</div>',
            unsafe_allow_html=True)

# Data freshness
last_data_date = pd.to_datetime(df['date'].max()).strftime('%B %d, %Y')
st.markdown(f"""
<div style="text-align:center; color:#6b7280; font-size:0.8rem; margin-top:-0.5rem; margin-bottom:1.5rem;">
    📅 Data last updated: {last_data_date}
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 🔍 Search for a Tablet")

col1, col2, col3 = st.columns([2,1,1])
with col1:
    search = st.text_input("Product name", placeholder="e.g. iPad, Samsung...")
with col2:
    brand_f = st.selectbox("Brand", ['All'] + [b.title() for b in KNOWN_BRANDS])
with col3:
    website_f = st.selectbox("Website", ['All'] + sorted([w.upper() for w in df['website'].unique()]))

# Filter
filtered = df.copy()
if search:
    filtered = filtered[filtered['name'].str.lower().str.contains(search.lower(), na=False)]
if brand_f != 'All':
    filtered = filtered[filtered['brand'] == brand_f.lower()]
if website_f != 'All':
    filtered = filtered[filtered['website'] == website_f.lower()]

# Get products
product_summary = filtered.groupby('product_key').agg(
    name=('name','first'), website=('website','first'),
    n_obs=('price','count')
).reset_index().sort_values('n_obs', ascending=False)

if product_summary.empty:
    st.warning("No products found.")
else:
    st.markdown(f"**{len(product_summary)} products found**")
    
    selected_key = st.selectbox(
        "Select product",
        options=product_summary['product_key'].tolist(),
        format_func=lambda k: (
            f"{product_summary[product_summary['product_key']==k]['name'].values[0]} | "
            f"{product_summary[product_summary['product_key']==k]['website'].values[0].upper()} "
            f"({product_summary[product_summary['product_key']==k]['n_obs'].values[0]} obs)"
        )
    )
    
    if selected_key:
        pdf = df[df['product_key'] == selected_key].copy()
        
        with st.spinner("Forecasting..."):
            try:
                result = forecast_product(pdf, days_ahead=7)
            except Exception as e:
                st.error(f"⚠️ Unable to forecast: {str(e)}")
                st.stop()
        
        st.markdown("---")
        
        # Model badge
        model_badge_class = 'badge-prophet' if result['model_type'] == 'Prophet' else 'badge-linear'
        model_badge = f'<span class="model-badge {model_badge_class}">{result["model_type"]}</span>'
        
        st.markdown(f"### 💰 Forecast Results {model_badge}", unsafe_allow_html=True)
        
        # Why this model?
        if result['model_type'] == 'Prophet':
            st.info(f"✨ Using **Prophet** (advanced AI model) — {result['n_obs']} observations available")
        else:
            st.info(f"⚡ Using **{result['model_type']}** (simple model) — {result['n_obs']} observations (Prophet needs 15+)")
        
        # Stats
        conf_color = {'High': '#00ff88', 'Medium': '#ffd166', 'Low': '#ff9966'}[result['confidence']]
        
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.markdown(f"""<div class="stat-card">
            <div class="stat-label">Current</div>
            <div class="stat-value" style="color:#00d4ff">EGP {result['last_price']:,.0f}</div>
        </div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class="stat-card">
            <div class="stat-label">7-Day</div>
            <div class="stat-value" style="color:#ffd166">EGP {result['forecast_prices'][-1]:,.0f}</div>
        </div>""", unsafe_allow_html=True)
        c3.markdown(f"""<div class="stat-card">
            <div class="stat-label">Average</div>
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
        
        # Buy signal
        signal_class = {'buy': 'signal-buy', 'wait': 'signal-wait', 'neutral': 'signal-neutral'}[result['signal']]
        
        st.markdown(f"""
        <div class="signal-card {signal_class}" style="padding:1.2rem 1.5rem; border-radius:16px; margin-bottom:1rem;">
            <div style="font-size:1.3rem; font-weight:800; margin-bottom:0.4rem;">
                {result['signal_text']}
            </div>
            <div style="font-size:0.95rem; color:#c0c4d0;">
                {result['signal_desc']}
            </div>
            <div style="margin-top:0.8rem; font-size:0.85rem; color:#8b92a5;">
                Forecast change: <strong style="color:{'#ff6b6b' if result['trend_pct'] > 0 else '#00ff88'}">
                {result['trend_pct']:+.1f}%</strong> over 7 days | 
                {result['n_obs']} observations | 
                Confidence: <strong style="color:{conf_color}">{result['confidence']}</strong> |
                Model: <strong>{result['model_type']}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Tomorrow
        tomorrow = result['forecast_prices'][0]
        tomorrow_diff = tomorrow - result['last_price']
        diff_color = "#ff6b6b" if tomorrow_diff > 0 else "#00ff88"
        
        st.markdown(f"""
        <div style="background:#111827; border:1px solid #1e2535; border-radius:14px;
                    padding:1rem 1.5rem; margin-bottom:1rem; display:flex;
                    justify-content:space-between; align-items:center;">
            <div>
                <div style="font-size:0.75rem; color:#6b7280; text-transform:uppercase;">Tomorrow</div>
                <div style="font-family:'Syne'; font-size:1.8rem; font-weight:800; color:#ffd166;">
                    EGP {tomorrow:,.0f}
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:0.75rem; color:#6b7280;">vs Today</div>
                <div style="font-family:'Syne'; font-size:1.3rem; font-weight:700; color:{diff_color};">
                    {'+' if tomorrow_diff > 0 else ''}EGP {abs(tomorrow_diff):,.0f}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Chart
        st.plotly_chart(chart_price_history(result), use_container_width=True)
        
        # 7-day table
        st.markdown("#### 📅 7-Day Forecast")
        forecast_df = pd.DataFrame({
            'Date': [d.strftime('%a, %b %d') for d in result['forecast_dates']],
            'Expected': [f"EGP {p:,.0f}" for p in result['forecast_prices']],
            'Low': [f"EGP {max(l,0):,.0f}" for l in result['forecast_lower']],
            'High': [f"EGP {u:,.0f}" for u in result['forecast_upper']],
        })
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        
        # Product link
        url = pdf['URL'].iloc[-1]
        if url and str(url) != 'nan':
            st.markdown(f'[🔗 View Product on {pdf["website"].iloc[0].upper()}]({url})')
