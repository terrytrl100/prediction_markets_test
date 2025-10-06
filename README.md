Polymarket Liquidity Provider (LP) Data Fetcher

**Overview**
This script is a Python tool for analysing Polymarket prediction markets, specifically designed for Liquidity Providers (LPs). 
It fetches real-time market data via the Polymarket ClobClient API, calculates estimated LP rewards per $100 provided (using geometric and simple means), computes annualized volatility across time windows (1h to 30d), and outputs results to CSV files or Google Sheets. 
Markets are ranked by reward potential while factoring in risk (volatility/reward ratio).

Key outputs:

(a) Top markets by gm_reward_per_100 (daily estimated earnings per $100 LP).
(b) Volatility metrics for risk assessment.
(c) Filtered views (e.g., low-vol markets).

This programme is ideal for identifying high-yield, low-risk LP opportunities in events like earnings bets or geopolitics.

**Features**.

(a) Market Sampling: Pulls all active Polymarket markets with rewards enabled.
(b) Order Book Analysis: Simulates LP rewards based on bid/ask depth near midpoint.
(c) Volatility Calculation: Annualized std dev of log returns from 1m price history.
(d) Risk-Adjusted Ranking: Composite scores balancing rewards, vol, and price balance.
(e) Output Options: Local CSVs (data/all_markets.csv, etc.) or Google Sheets integration.
(f) Parallel Processing: Fast execution with threading for order books and volatility.

**Prerequisites**

Python 3.8+.
A Polymarket wallet (on Polygon) with USDC for LP (optional for data fetch).
GitHub account for repo access (if forking).

Setup
1. Clone the Repo
bashgit clone https://github.com/terrytrl100/prediction_markets_test.git
cd prediction_markets_test

3. Create Virtual Environment
bashpython -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

4. Install Dependencies
bashpip install -r requirements.txt  # Assumes you add one; see below for manual
Manual install (core libs):
bashpip install py-clob-client gspread gspread-dataframe pandas numpy requests python-dotenv web3

5. Configure Environment Variables (.env)
Create a .env file in the root directory (ignored by Git). Add your Polymarket wallet details:
textPK=your_polygon_private_key_here  # Hex string, e.g., 0xabc123... (no 0x prefix if raw)
SPREADSHEET_URL=https://docs.google.com/spreadsheets/d/your_sheet_id/edit  # Optional, for Sheets output

PK: Your Polygon private key (from wallet like MetaMask). Warning: This enables API/trading—use a dev wallet with minimal funds. Never commit .env!
SPREADSHEET_URL: Google Sheets link for output (e.g., "All Markets" tab).

5. Google Sheets Setup (credentials.json)
For writing to Sheets (optional; falls back to read-only CSV):

- Go to Google Cloud Console.
- Create a project > Enable Google Sheets/Drive APIs.
- Create a Service Account (IAM & Admin > Service Accounts > Create).
- Download the JSON key file as credentials.json in root.
- Share your Sheet with the service account email (from JSON).
- Security: Add credentials.json to .gitignore—don't commit!

**Usage**
Run the Script
bashpython data_updater.py

Fetches ~1000+ markets.

**Outputs:
**
data/all_markets.csv: Full ranked list.
data/volatility_markets.csv: Low-vol (<20 sum) subset.
data/full_markets.csv: Raw order book data.
Console: Top 10 by rewards + progress logs.


Runtime: 2–5 min (depends on API rate limits).

Example Output
textTop 10 Markets (by gm_reward_per_100):
[Table of markets with rewards, vol, odds...]
Trading/LP Integration (Advanced)

Use approveContracts() to approve USDC/CTF (run once).
market_action(marketId, 'BUY'/'SELL', price, size): Place orders.
get_position(marketId): Check holdings value.

Dependencies
Add to requirements.txt:
textpy-clob-client==0.25.0
gspread==6.1.0
gspread-dataframe==15.0.0
pandas==2.2.2
numpy==1.26.4
requests==2.32.3
python-dotenv==1.0.1
web3==6.15.1
Contributing

Fork the repo.
Create a feature branch (git checkout -b feat/amazing-feature).
Commit changes (git commit -m 'Add some feature').
Push (git push origin feat/amazing-feature).
Open a Pull Request.
