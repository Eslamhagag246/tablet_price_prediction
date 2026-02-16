# tablet_price_prediction
📊 Tablet Price Tracker
A data-driven web application that tracks tablet prices across major Egyptian e-commerce websites, forecasts prices for the next 7 days, and helps users decide the best time to buy.

🌐 Live Demo
👉 Open the App

💡 Project Idea
The goal of this project is to help users make smarter purchasing decisions by answering two key questions:

How has this tablet's price changed over time?
Is now a good time to buy, or should I wait?

To answer these questions, price data was scraped from 5 major Egyptian e-commerce websites over a period of ~2 months, covering 11 tablet brands across 269 unique product listings.

🔍 Features
🔎 Product Forecast Tab

Search for any tablet by name, brand, or website
View current price, average price, min/max range
Get a 7-day price forecast with confidence band
See tomorrow's expected price and how it compares to today
Receive a smart buy signal:

🟢 Good Time to Buy
🔴 Wait — Price May Drop
🔴 Price Rising — Buy Soon
🟡 Price is Stable


Interactive price history chart with forecast overlay
Price distribution and daily volatility charts
Full 7-day forecast table with low/high price estimates
Direct product link to the website

📊 Data Overview Tab

Dataset statistics (total records, unique products, date range, websites tracked)
Average price trends by brand over time
Price range comparison across websites (box plot)
Distribution of price observations per product

📈 Market Insights Tab

Top 5 biggest price drops since tracking started
Top 5 biggest price rises since tracking started
Horizontal bar chart of all price changes across products


🛠️ How It Works
Data Collection
Price data was scraped from 5 websites:

Jumia
BTECH
Dream2000
2B
Dubaiphone

Preprocessing

Cleaned and normalized price, brand, website, and timestamp fields
Averaged duplicate prices scraped on the same day for the same product
Created a unique product identity per listing (product name + website)
Engineered time-based features: day index, rolling average, volatility, % change

Forecasting Model
A per-product time-series regression approach:

Linear Regression for products with fewer than 10 observations
Polynomial Regression (degree 2) for products with 10+ observations
Forecast clipped to realistic price bounds (±50% of historical range)
Confidence band based on mean absolute error of the fit
Confidence level assigned based on number of available data points (Low / Medium / High)

Buy Signal Logic
The buy signal is calculated from two factors:

Current price vs historical average — is it currently cheap or expensive?
Forecasted trend — is the price expected to go up or down?


📦 Tech Stack
ToolPurposePythonCore languagePandasData manipulationNumPyNumerical operationsScikit-learnLinear & Polynomial RegressionPlotlyInteractive chartsStreamlitWeb application frameworkGitHubVersion control & deploymentStreamlit CloudFree app hosting

🗂️ Project Structure
tablet_price_prediction/
│
├── streamlit_app.py          # Main Streamlit web application
├── tablets_cleaned_clean.csv # Cleaned scraped price dataset
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation

⚠️ Limitations

Data covers ~56 days — longer history would improve forecast accuracy
Most products have fewer than 10 price observations, so forecasts are based on short trends
Confidence is shown transparently (Low / Medium / High) so users know how much to rely on the forecast
Prices are in Egyptian Pounds (EGP)


👨‍💻 Author
Built by Eslamhagag246
