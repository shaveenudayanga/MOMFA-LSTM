from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import os
from dotenv import load_dotenv

# 1. Load the credentials from your .env file into the script
load_dotenv()

# Ensure the data directory exists
os.makedirs('data', exist_ok=True)

# 2. Fetch the credentials (these will no longer be empty)
username = os.getenv('TRADINGVIEW_USERNAME')
password = os.getenv('TRADINGVIEW_PASSWORD')

# 3. Initialize with authentication
tv = TvDatafeed(username=username, password=password)

tickers = ["JKH.N0000", "COMB.N0000", "DIAL.N0000", "HNB.N0000", "LOLC.N0000", "SAMP.N0000", "NTB.N0000", "HHL.N0000", "DIST.N0000", "HAYL.N0000"]

for ticker in tickers:
    print(f"Fetching {ticker}...")
    
    # 4. Download historical data
    data = tv.get_hist(
        symbol=ticker, 
        exchange='CSELK', 
        interval=Interval.in_daily, 
        n_bars=1500
    )
    
    # 5. Exporting Data
    if data is not None and not data.empty:
        # Slice the dataframe to grab exactly 2021-01-01 to 2025-12-31
        data = data[(data.index >= '2021-01-01') & (data.index <= '2025-12-31')]
        
        # Save to CSV
        file_path = f"data/{ticker}_data.csv"
        data.to_csv(file_path)
        print(f"Successfully downloaded {ticker}: {len(data)} rows")
    else:
        print(f"Failed to download {ticker}")