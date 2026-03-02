import pandas as pd
from datetime import datetime
import asyncio
from playwright.async_api import async_playwright
import sys
import random
import re
from concurrent.futures import ThreadPoolExecutor

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

CSV_FILE = 'tablets_full_continuous_series.csv'

# Correct selectors for all websites
SELECTORS = {
    'jumia': {
        'price': ['div.-paxs span', 'span[data-price]', '.prc', 'div.-fs0.-tal span'],
    },
    'btech': {
        'price': ['span.text-medium'],
    },
    'b.tech': {
        'price': ['span.text-medium'],
    },
    'dream2000': {
        'price': ['span.price'],
    },
    'dream 2000': {
        'price': ['span.price'],
    },
    '2b': {
        'price': ['span.price'],
    },
    'dubaiphone': {
        'price': ['span.woocommerce-Price-amount.amount bdi', 'bdi'],
    }
}

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
]

STEALTH_JS = """
() => {
    Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
    Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3]});
    window.chrome = {runtime: {}};
}
"""

def get_selectors(website):
    website_lower = website.lower().strip().replace(' ', '')
    if website_lower in SELECTORS:
        return SELECTORS[website_lower]['price']
    for key in SELECTORS.keys():
        clean_key = key.replace(' ', '')
        if clean_key in website_lower or website_lower in clean_key:
            return SELECTORS[key]['price']
    return ['span.price', '.price', 'bdi']

def clean_price(price_text):
    if not price_text:
        return None
    price_text = price_text.strip()
    if price_text.lower() in ['english', 'arabic', 'french', 'language']:
        return None
    if not any(c.isdigit() for c in price_text):
        return None
    price_text = re.sub(r'From.*', '', price_text)
    price_text = re.sub(r'EGP\s*', '', price_text, flags=re.IGNORECASE)
    price_text = re.sub(r'[^\d,.]', '', price_text)
    price_text = price_text.strip()
    if price_text and any(c.isdigit() for c in price_text):
        return f"EGP {price_text}"
    return None

async def scrape_product_fast(context, url, website):
    """Optimized scraper - 3x faster"""
    page = None
    try:
        page = await context.new_page()
        await page.add_init_script(STEALTH_JS)
        await asyncio.sleep(random.uniform(0.5, 1))  
        await page.goto(url, 
                       wait_until='domcontentloaded',  
                       timeout=30000) 
        await asyncio.sleep(random.uniform(1, 1.5)) 
        price_selectors = get_selectors(website)
        price = None
        
        for sel in price_selectors:
            try:
                elems = await page.query_selector_all(sel)
                for elem in elems:
                    text = await elem.inner_text()
                    cleaned = clean_price(text)
                    if cleaned:
                        price = cleaned
                        break
                if price:
                    break
            except:
                pass
        if not price and website.lower() in ['dream2000', 'dream 2000', '2b']:
            try:
                price_amount = await page.evaluate(
                    "document.querySelector('[data-price-amount]')?.getAttribute('data-price-amount')"
                )
                if price_amount:
                    price = f"EGP {price_amount}"
            except:
                pass
        
        await page.close()
        return {'price': price, 'stock': 'In stock'} if price else None
        
    except:
        if page:
            try:
                await page.close()
            except:
                pass
        return None

def get_last_price(df, product_name, website):
    """Get last known price for a failed product"""
    product_data = df[(df['name'] == product_name) & (df['website'] == website)]
    if not product_data.empty:
        # Get most recent price
        product_data = product_data.sort_values('timestamp', ascending=False)
        return product_data.iloc[0]['price']
    return None

async def main():
    print("="*80)
    print("🚀 OPTIMIZED FINAL SCRAPER")
    print("="*80)
    
    # Load data
    try:
        df = pd.read_csv(CSV_FILE)
        print(f"✅ Loaded {len(df)} records")
    except:
        print(f"❌ File not found")
        sys.exit(1)
    
    # Remove duplicates
    print(f"\n🔍 Removing duplicates...")
    before = len(df)
    products = df[['name','website','URL','brand','ram_gb','storage_gb']]\
        .drop_duplicates(subset=['name','website'], keep='first')
    after = len(products)
    
    print(f"   Removed: {before - after} duplicates")
    print(f"   Total unique products: {after}")
    
    print(f"\n📊 Starting scrape...")
    print(f"🚀 Started at {datetime.now().strftime('%H:%M:%S')}")
    print("─"*80)
    
    new_data = []
    success = fail = retry_success = 0
    website_stats = {}
    
    start_time = datetime.now()
    
    async with async_playwright() as p:

        browser = None
        context = None

        for idx, row in products.iterrows():

            # 🔥 Restart browser every 100 products
            if idx % 100 == 0:
                if browser:
                    try:
                        await context.close()
                        await browser.close()
                    except:
                        pass

                browser = await p.chromium.launch(
                    headless=True,
                    args=[
                        '--disable-blink-features=AutomationControlled',
                        '--no-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-gpu',
                        '--disable-setuid-sandbox'
                    ]
                )

                context = await browser.new_context(
                    user_agent=random.choice(USER_AGENTS),
                    viewport={'width': 1920, 'height': 1080}
                )
        
        # Process products
        for idx, row in products.iterrows():
            current = idx + 1  # ✅ FIXED COUNTER
            total = len(products)
            website = row['website']
            product_name = row['name']
            
            # ✅ Progress bar with percentage
            progress = (current / total) * 100
            print(f"[{current}/{total}] ({progress:.1f}%) {product_name[:35]:<35} | {website.upper():<10}", 
                  end=" ", flush=True)
            
            result = await scrape_product_fast(context, row['URL'], website)
            
            if result:
                # Success!
                new_data.append({
                    'name': product_name,
                    'price': result['price'],
                    'stock': result['stock'],
                    'photo': '',
                    'URL': row['URL'],
                    'website': website,
                    'timestamp': datetime.now().strftime('%m/%d/%Y'),
                    'brand': row['brand'],
                    'ram_gb': row['ram_gb'],
                    'storage_gb': row['storage_gb'],
                    'product_key': ''
                })
                print(f"✅ {result['price']}")
                success += 1
                website_stats[website] = website_stats.get(website, {'success': 0, 'fail': 0})
                website_stats[website]['success'] += 1
                
            else:
                # ✅ RETRY ONCE
                print(f"🔄", end=" ", flush=True)
                await asyncio.sleep(random.uniform(2, 3))
                
                result = await scrape_product_fast(context, row['URL'], website)
                
                if result:
                    # Retry success!
                    new_data.append({
                        'name': product_name,
                        'price': result['price'],
                        'stock': result['stock'],
                        'photo': '',
                        'URL': row['URL'],
                        'website': website,
                        'timestamp': datetime.now().strftime('%m/%d/%Y'),
                        'brand': row['brand'],
                        'ram_gb': row['ram_gb'],
                        'storage_gb': row['storage_gb'],
                        'product_key': ''
                    })
                    print(f"✅ {result['price']}")
                    retry_success += 1
                    website_stats[website] = website_stats.get(website, {'success': 0, 'fail': 0})
                    website_stats[website]['success'] += 1
                else:
                    # ✅ FAILED → Use last known price with today's date
                    last_price = get_last_price(df, product_name, website)
                    
                    if last_price:
                        new_data.append({
                            'name': product_name,
                            'price': last_price, 
                            'stock': 'In stock',
                            'photo': '',
                            'URL': row['URL'],
                            'website': website,
                            'timestamp': datetime.now().strftime('%m/%d/%Y'),
                            'brand': row['brand'],
                            'ram_gb': row['ram_gb'],
                            'storage_gb': row['storage_gb'],
                            'product_key': ''
                        })
                        print(f"♻️  {last_price} (last known)")
                    else:
                        print(f"❌ Failed (no history)")
                    
                    fail += 1
                    website_stats[website] = website_stats.get(website, {'success': 0, 'fail': 0})
                    website_stats[website]['fail'] += 1
        
        await context.close()
        await browser.close()
    
    # Calculate time
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    avg_time = duration / len(products) if len(products) > 0 else 0
    
    # Process results
    if not new_data:
        print("\n⚠️  No data")
        return
    
    print(f"\n🧹 Processing {len(new_data)} records...")
    
    new_df = pd.DataFrame(new_data)
    new_df['product_key'] = (
        new_df['name'].str.lower().str.strip() + ' ' +
        new_df['website'].str.lower() + ' ' +
        new_df['ram_gb'].astype(str) + ' ' +
        new_df['storage_gb'].astype(str)
    )
    
    # Merge
    combined = pd.concat([df, new_df], ignore_index=True)
    
    # Remove same-day duplicates
    combined['date_check'] = pd.to_datetime(
        combined['timestamp'], format='%m/%d/%Y', errors='coerce'
    ).dt.date
    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=['name', 'website', 'date_check','price'], keep='last')
    after_dedup = len(combined)
    combined = combined.drop('date_check', axis=1)
    
    # Sort
    combined['ts_sort'] = pd.to_datetime(combined['timestamp'], format='%m/%d/%Y', errors='coerce')
    combined = combined.sort_values('ts_sort')
    combined = combined.drop('ts_sort', axis=1)
    
    # Save
    combined.to_csv(CSV_FILE, index=False)
    
    # Summary
    print("\n" + "="*80)
    print("📊 SCRAPING SUMMARY")
    print("="*80)
    print(f"Products scraped:     {len(products)}")
    print(f"Fresh prices:         {success} ✅")
    print(f"Retry successes:      {retry_success} 🔄")
    print(f"Used last price:      {fail - sum(1 for d in new_data if d['price'] is None)} ♻️")
    print(f"Total failures:       {sum(1 for d in new_data if d['price'] is None)} ❌")
    print(f"\nSuccess rate:         {(success + retry_success)/len(products)*100:.1f}%")
    print(f"\nNew rows added:       {len(new_df)}")
    print(f"Duplicates removed:   {before_dedup - after_dedup}")
    print(f"Total rows in CSV:    {len(combined)}")
    
    # ✅ PERFORMANCE STATS
    print(f"\n⚡ PERFORMANCE:")
    print(f"   Total time:        {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"   Avg per product:   {avg_time:.2f} seconds")
    print(f"   Speed:             {len(products)/duration*60:.1f} products/minute")
    
    # Website breakdown
    print(f"\n📊 BY WEBSITE:")
    print("─"*80)
    for website in sorted(website_stats.keys()):
        stats = website_stats[website]
        total_site = stats['success'] + stats['fail']
        rate = stats['success'] / total_site * 100 if total_site > 0 else 0
        print(f"   {website.upper():<12} {stats['success']}/{total_site} ({rate:.1f}%)")
    
    print("="*80)
    print("✅ Done! Upload CSV to GitHub.")

if __name__ == "__main__":
    asyncio.run(main())