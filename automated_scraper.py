import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
import sys

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
CSV_FILE = 'tablets_full_continuous_series.csv'
REQUEST_TIMEOUT = 10  # seconds
DELAY_BETWEEN_REQUESTS = 2  # seconds (be nice to servers)

# Website-specific selectors
SELECTORS = {
    'jumia': {
        'price': ['.prc', '-fs', '.price'],
        'stock': ['.stock', '.in-stock']
    },
    'btech': {
        'price': ['.price', '.product-price', '.current-price'],
        'stock': ['.stock', '.availability']
    },
    'dream2000': {
        'price': ['.price', '.product-price'],
        'stock': ['.stock']
    },
    '2b': {
        'price': ['.price', '.product-price'],
        'stock': ['.stock']
    },
    'dubaiphone': {
        'price': ['.price', '.woocommerce-Price-amount'],
        'stock': ['.stock']
    }
}

# ─────────────────────────────────────────
# 1. LOAD EXISTING DATA
# ─────────────────────────────────────────
print("=" * 60)
print("🤖 AUTOMATED PRICE SCRAPER")
print("=" * 60)

try:
    existing_df = pd.read_csv(CSV_FILE)
    print(f"✅ Loaded {len(existing_df)} existing records")
except FileNotFoundError:
    print(f"❌ Error: {CSV_FILE} not found!")
    sys.exit(1)

# Get unique products we're tracking
products_to_track = existing_df[['name', 'website', 'URL', 'brand', 'ram_gb', 'storage_gb']].drop_duplicates(subset=['name', 'website'])
print(f"📊 Tracking {len(products_to_track)} unique products")

# ─────────────────────────────────────────
# 2. SCRAPING FUNCTIONS
# ─────────────────────────────────────────
def get_selector_for_website(website, field):
    """Get CSS selectors for a website"""
    website_lower = website.lower()
    for site_key, selectors in SELECTORS.items():
        if site_key in website_lower:
            return selectors.get(field, [])
    return []

def extract_price(soup, website):
    """Extract price from HTML"""
    price_selectors = get_selector_for_website(website, 'price')
    
    for selector in price_selectors:
        elem = soup.select_one(selector)
        if elem:
            return elem.text.strip()
    
    return None

def extract_stock(soup, website):
    """Extract stock status from HTML"""
    stock_selectors = get_selector_for_website(website, 'stock')
    
    for selector in stock_selectors:
        elem = soup.select_one(selector)
        if elem:
            text = elem.text.strip().lower()
            if 'in stock' in text or 'متوفر' in text:
                return 'In stock'
            elif 'out' in text or 'نفذ' in text:
                return 'Out of stock'
    
    return 'In stock'  # Default
session = requests.Session()
headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",}
def scrape_product(url, website):
    """Scrape a single product"""
    try:
        response = session.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        price = extract_price(soup, website)
        stock = extract_stock(soup, website)

        if price:
            return {'price': price, 'stock': stock}
        else:
            return None
            
    except requests.exceptions.Timeout:
        print(f"    ⏱️  Timeout")
        return None
    except requests.exceptions.RequestException as e:
        print(f"    ❌ Error: {str(e)[:50]}")
        return None
    except Exception as e:
        print(f"    ❌ Unexpected error: {str(e)[:50]}")
        return None

# ─────────────────────────────────────────
# 3. SCRAPE ALL TRACKED PRODUCTS
# ─────────────────────────────────────────
print(f"\n🔍 Starting scrape at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("─" * 60)

new_data = []
success_count = 0
fail_count = 0

for idx, row in products_to_track.iterrows():
    product_name = row['name']
    website = row['website']
    url = row['URL']
    
    website_safe = str(website) if pd.notna(website) else "UNKNOWN"
    print(f"[{idx+1}/{len(products_to_track)}] {product_name[:40]:<40} | {website_safe.upper():<10}", end=" ")
    
    # Scrape
    result = scrape_product(url, website)
    
    if result:
        new_data.append({
            'name': product_name,
            'price': result['price'],
            'stock': result['stock'],
            'photo': '',  # Keep empty for now
            'URL': url,
            'website': website,
            'timestamp': datetime.now().strftime('%m/%d/%Y'),
            'brand': row['brand'],
            'ram_gb': row['ram_gb'],
            'storage_gb': row['storage_gb'],
            'product_key': ''  # Will be recalculated
        })
        print(f"✅ {result['price']}")
        success_count += 1
    else:
        print(f"❌ Failed")
        fail_count += 1
    
    # Delay between requests
    time.sleep(DELAY_BETWEEN_REQUESTS)

print("─" * 60)
print(f"✅ Success: {success_count} products")
print(f"❌ Failed:  {fail_count} products")

# ─────────────────────────────────────────
# 4. CLEAN NEW DATA
# ─────────────────────────────────────────
if len(new_data) == 0:
    print("\n⚠️  No new data scraped. Exiting without updating CSV.")
    sys.exit(0)

print(f"\n🧹 Cleaning {len(new_data)} new records...")

new_df = pd.DataFrame(new_data)

new_df['name'] = new_df['name'].astype(str)
new_df['name'] = new_df['name'].str.replace(',', '', regex=False)
new_df['name'] = new_df['name'].str.replace('/', '', regex=False)
new_df['name'] = new_df['name'].str.replace(r'(?i)2 years warranty', '', regex=True)
new_df['name'] = new_df['name'].str.replace(r'(?i)tax paid', '', regex=True)
colors = ['cosmic orange', 'deep', 'lavender', 'teal', 'sage']
for color in colors:
    new_df['name'] = new_df['name'].str.replace(fr'(?i){color}', '', regex=True)

# Remove multiple spaces
new_df['name'] = new_df['name'].str.replace(r'\s+', ' ', regex=True).str.strip()

# Clean prices
new_df['price'] = new_df['price'].astype(str)
new_df['price'] = new_df['price'].str.replace('EGP', '', regex=False)
new_df['price'] = new_df['price'].str.replace('E£', '', regex=False)
new_df['price'] = new_df['price'].str.replace('ج.م', '', regex=False)
new_df['price'] = new_df['price'].str.replace(',', '', regex=False)
new_df['price'] = new_df['price'].str.strip()

# Add "EGP" prefix
new_df['price'] = 'EGP ' + new_df['price']

# Recreate product_key
new_df['product_key'] = (
    new_df['name'].str.lower().str.strip() + ' ' +
    new_df['website'].str.lower() + ' ' +
    new_df['ram_gb'].astype(str) + ' ' +
    new_df['storage_gb'].astype(str)
)

print("✅ Data cleaned")

# ─────────────────────────────────────────
# 5. MERGE WITH EXISTING DATA
# ─────────────────────────────────────────
print("\n📊 Merging with existing data...")

# Combine
combined_df = pd.concat([existing_df, new_df], ignore_index=True)

# Remove duplicates (same product on same day)
combined_df['date_temp'] = pd.to_datetime(combined_df['timestamp'], format='%m/%d/%Y').dt.date
combined_df = combined_df.drop_duplicates(subset=['name', 'website', 'date_temp'], keep='last')
combined_df = combined_df.drop('date_temp', axis=1)

# Sort by timestamp
combined_df['ts_temp'] = pd.to_datetime(combined_df['timestamp'], format='%m/%d/%Y')
combined_df = combined_df.sort_values('ts_temp')
combined_df = combined_df.drop('ts_temp', axis=1)

print(f"  Before: {len(existing_df)} rows")
print(f"  After:  {len(combined_df)} rows")
print(f"  Added:  {len(combined_df) - len(existing_df)} new rows")

# ─────────────────────────────────────────
# 6. SAVE
# ─────────────────────────────────────────
combined_df.to_csv(CSV_FILE, index=False)
print(f"\n✅ Saved to {CSV_FILE}")

# ─────────────────────────────────────────
# 7. SUMMARY
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("📊 SCRAPING SUMMARY")
print("=" * 60)
print(f"Products scraped:     {success_count}/{len(products_to_track)}")
print(f"Success rate:         {success_count/len(products_to_track)*100:.1f}%")
print(f"New rows added:       {len(combined_df) - len(existing_df)}")
print(f"Total rows in CSV:    {len(combined_df)}")
print(f"Unique products:      {combined_df['product_key'].nunique()}")
print("=" * 60)
print("✅ Done! CSV will be auto-committed by GitHub Actions.")
