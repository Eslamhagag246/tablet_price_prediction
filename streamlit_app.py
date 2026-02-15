import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Tablet Price Predictor",
    page_icon="📱",
    layout="centered"
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark background */
.stApp {
    background: #0b0f1a;
    color: #e8eaf0;
}

/* Header */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00d4ff 0%, #7b5cf0 50%, #ff6b9d 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    color: #6b7280;
    margin-bottom: 2rem;
    font-weight: 300;
    letter-spacing: 0.05em;
}

/* Cards */
.result-card {
    background: #141927;
    border: 1px solid #1e2535;
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s;
}
.result-card:hover {
    border-color: #00d4ff44;
}
.result-card.best {
    border: 1.5px solid #00d4ff55;
    background: linear-gradient(135deg, #0d1f2d 0%, #141927 100%);
}
.website-name {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #e8eaf0;
}
.product-name {
    font-size: 0.85rem;
    color: #8b92a5;
    margin: 0.2rem 0;
}
.price-tag {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #00d4ff;
}
.price-tag.expensive {
    color: #ff6b6b;
}
.url-link {
    font-size: 0.78rem;
    color: #7b5cf0;
    word-break: break-all;
    text-decoration: none;
}
.badge-best {
    display: inline-block;
    background: linear-gradient(135deg, #00d4ff22, #7b5cf022);
    border: 1px solid #00d4ff55;
    color: #00d4ff;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 20px;
    margin-left: 8px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.saving-box {
    background: linear-gradient(135deg, #0d2b1d, #0b1a12);
    border: 1px solid #00ff8833;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin-top: 1rem;
    color: #00ff88;
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 600;
}
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.metric-box {
    flex: 1;
    background: #141927;
    border: 1px solid #1e2535;
    border-radius: 12px;
    padding: 0.9rem 1.2rem;
    text-align: center;
}
.metric-label {
    font-size: 0.72rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #7b5cf0;
}

/* Streamlit widgets overrides */
.stSelectbox label, .stSlider label, .stNumberInput label {
    color: #8b92a5 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
div[data-testid="stSelectbox"] > div {
    background: #141927 !important;
    border: 1px solid #1e2535 !important;
    border-radius: 10px !important;
    color: #e8eaf0 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #7b5cf0, #00d4ff) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
    letter-spacing: 0.05em !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover {
    opacity: 0.85 !important;
}
hr {
    border-color: #1e2535 !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# LOAD & TRAIN MODEL (cached)
# ─────────────────────────────────────────
@st.cache_resource(show_spinner="Training model on your data...")
def load_and_train():
    df = pd.read_csv('tablets_cleaned_clean.csv')

    df['price']   = df['price'].str.replace('EGP', '', regex=False)\
                               .str.replace(',', '', regex=False)\
                               .str.strip().astype(float)
    df['brand']   = df['brand'].str.lower().str.strip()
    df['website'] = df['website'].str.lower().str.strip()
    df['stock']   = df['stock'].str.lower().str.strip()

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['year']      = df['timestamp'].dt.year
    df['month']     = df['timestamp'].dt.month
    df['day']       = df['timestamp'].dt.day
    df['dayofweek'] = df['timestamp'].dt.dayofweek

    df['ram_gb']            = df['ram_gb'].fillna(df['ram_gb'].median())
    df['ram_storage_ratio'] = df['ram_gb'] / (df['storage_gb'] + 1)
    df['storage_per_egp']   = df['storage_gb'] / df['price']

    le_brand   = LabelEncoder()
    le_website = LabelEncoder()
    df['brand_enc']   = le_brand.fit_transform(df['brand'])
    df['website_enc'] = le_website.fit_transform(df['website'])

    FEATURES = ['ram_gb','storage_gb','brand_enc','website_enc',
                'year','month','day','dayofweek',
                'ram_storage_ratio','storage_per_egp']

    X = df[FEATURES]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=300, max_features='sqrt', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    gb = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                                    subsample=0.8, min_samples_split=4, min_samples_leaf=2, random_state=42)
    gb.fit(X_train, y_train)

    r2_rf  = r2_score(y_test, rf.predict(X_test))
    r2_gb  = r2_score(y_test, gb.predict(X_test))
    mae_rf = mean_absolute_error(y_test, rf.predict(X_test))
    mae_gb = mean_absolute_error(y_test, gb.predict(X_test))

    best_model      = gb if r2_gb >= r2_rf else rf
    best_model_name = "Gradient Boosting" if r2_gb >= r2_rf else "Random Forest"

    return df, le_brand, le_website, FEATURES, best_model, best_model_name


def find_product(df, brand, ram, storage, website):
    subset = df[(df['brand']==brand) & (df['ram_gb']==ram) &
                (df['storage_gb']==storage) & (df['website']==website)]
    if subset.empty:
        subset = df[(df['brand']==brand) & (df['website']==website)]
    if subset.empty:
        return 'N/A', '#'
    row = subset.sort_values('timestamp', ascending=False).iloc[0]
    return row['name'].strip(), row['URL']


def predict_prices(df, le_brand, le_website, FEATURES, model, brand, ram, storage):
    today = datetime.today()
    year, month, day, dow = today.year, today.month, today.day, today.weekday()
    brand_enc = le_brand.transform([brand])[0]
    results = []
    for ws in le_website.classes_:
        ws_enc  = le_website.transform([ws])[0]
        avg_spe = df[df['website']==ws]['storage_per_egp'].mean()
        row = pd.DataFrame([[ram, storage, brand_enc, ws_enc,
                             year, month, day, dow,
                             ram/(storage+1), avg_spe]], columns=FEATURES)
        pred = model.predict(row)[0]
        name, url = find_product(df, brand, ram, storage, ws)
        results.append({'website': ws.upper(), 'name': name,
                        'price': round(pred, 2), 'url': url})
    return sorted(results, key=lambda x: x['price'])


# ─────────────────────────────────────────
# LOAD DATA & MODEL
# ─────────────────────────────────────────
df, le_brand, le_website, FEATURES, model, model_name = load_and_train()
KNOWN_BRANDS = sorted(df['brand'].unique().tolist())

# ─────────────────────────────────────────
# UI — HEADER
# ─────────────────────────────────────────
st.markdown('<div class="hero-title">📱 Tablet Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Find the best deal across all Egyptian e-commerce websites</div>', unsafe_allow_html=True)



# ─────────────────────────────────────────
# UI — INPUT FORM
# ─────────────────────────────────────────
st.markdown("### 🔍 Describe Your Tablet")

col1, col2, col3 = st.columns(3)

with col1:
    brand = st.selectbox("Brand", options=[b.title() for b in KNOWN_BRANDS])

with col2:
    ram = st.selectbox("RAM (GB)", options=[2, 3, 4, 6, 8, 12, 16], index=4)

with col3:
    storage = st.selectbox("Storage (GB)", options=[32, 64, 128, 256, 512], index=2)

st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🔮 Predict Prices Across All Websites")

# ─────────────────────────────────────────
# UI — RESULTS
# ─────────────────────────────────────────
if predict_btn:
    brand_lower = brand.lower()

    if brand_lower not in KNOWN_BRANDS:
        st.error("Brand not found in training data.")
    else:
        with st.spinner("Predicting..."):
            results = predict_prices(df, le_brand, le_website, FEATURES,
                                     model, brand_lower, ram, storage)

        best  = results[0]
        worst = results[-1]
        saving = worst['price'] - best['price']

        st.markdown("---")
        st.markdown(f"### 💰 Results for **{brand}** | {ram}GB RAM | {storage}GB Storage")
        st.markdown("<br>", unsafe_allow_html=True)

        medals = ['🥇', '🥈', '🥉']

        for i, r in enumerate(results):
            import html as html_lib
            medal     = medals[i] if i < 3 else f"{i+1}."
            is_best   = i == 0
            is_worst  = i == len(results) - 1
            card_cls  = "result-card best" if is_best else "result-card"
            badge     = '<span class="badge-best">Best Deal</span>' if is_best else ""
            price_cls = "price-tag expensive" if is_worst else "price-tag"

            # Escape special characters to prevent HTML breaking
            safe_name = html_lib.escape(str(r['name']))
            safe_url  = html_lib.escape(str(r['url'])) if r['url'] != '#' else '#'

            url_html = (
                f'<a href="{safe_url}" target="_blank" class="url-link">🔗 {safe_url}</a>'
                if r['url'] != '#'
                else '<span class="url-link" style="color:#444">URL not available</span>'
            )

            st.markdown(f"""
            <div class="{card_cls}">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
                    <div>
                        <span style="font-size:1.4rem">{medal}</span>
                        <span class="website-name">&nbsp;{r['website']}</span>
                        {badge}
                    </div>
                    <div class="{price_cls}">EGP {r['price']:,.2f}</div>
                </div>
                <div class="product-name" style="margin-top:0.5rem;">📦 {safe_name}</div>
                <div style="margin-top:0.4rem;">{url_html}</div>
            </div>
            """, unsafe_allow_html=True)

        # Savings summary
        st.markdown(f"""
        <div class="saving-box">
            💸 Buy from <strong>{best['website']}</strong> and save
            <strong>EGP {saving:,.2f}</strong> vs {worst['website']}
        </div>
        """, unsafe_allow_html=True)
