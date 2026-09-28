from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import os
from dotenv import load_dotenv

# 1. Load the credentials from your .env file into the script
load_dotenv()

# Ensure the specific data/ASPI directory exists
output_dir = os.path.join('data', 'ASPI')
os.makedirs(output_dir, exist_ok=True)

# 2. Fetch the credentials
username = os.getenv('TRADINGVIEW_USERNAME')
password = os.getenv('TRADINGVIEW_PASSWORD')

# 3. Initialize with authentication
tv = TvDatafeed(username=username, password=password)

# Symbol for All Share Price Index on CSE
ticker = "ASI"

print(f"Fetching {ticker}...")

# 4. Download historical data
# 15 years of daily data requires approx. 3750 bars (250 trading days/year)
# Setting n_bars to 4500 to guarantee full coverage back to Jan 1, 2010
data = tv.get_hist(
    symbol=ticker, 
    exchange='CSELK', 
    interval=Interval.in_daily, 
    n_bars=4500
)

# 5. Exporting Data
if data is not None and not data.empty:
    # Slice the dataframe to grab exactly 2010-01-01 to 2025-12-31
    data = data[(data.index >= '2010-01-01') & (data.index <= '2025-12-31')]
    
    # Save to CSV inside the data/ASPI directory
    file_path = os.path.join(output_dir, f"{ticker}_data.csv")
    data.to_csv(file_path)
    print(f"Successfully downloaded {ticker}: {len(data)} rows saved to {file_path}")
else:
    print(f"Failed to download {ticker}")