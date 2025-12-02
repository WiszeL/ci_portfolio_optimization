import pandas as pd
import yfinance as yf
from datetime import date


def download_data(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    print("========== Downloading Data ==========")
    print(f"Downloading data for {len(tickers)} stocks from {start} to {end}...")

    # Download raw data from Yahoo Finance
    # auto_adjust=True handles stock splits and dividends automatically
    # This ensures we work with Adjusted Close prices as per best practices
    raw_data = yf.download(tickers, start=start, end=end, auto_adjust=True)

    # Type Guard: Fixes Pylance "None is not subscriptable" or empty errors
    if raw_data is None or raw_data.empty:
        print("Error: No data found or download failed.")
        return pd.DataFrame()

    # Access the 'Close' column
    # yfinance typically returns a MultiIndex or a DataFrame with tickers as columns
    try:
        close_data = raw_data["Close"]
    except KeyError:
        print("Error: 'Close' column not found in downloaded data.")
        return pd.DataFrame()

    # Explicitly convert Series to DataFrame to ensure consistency
    # This happens if the user only requests a single ticker
    if isinstance(close_data, pd.Series):
        data = close_data.to_frame()
    else:
        data = close_data

    # Data Cleaning: Forward Fill (Imputation)
    # Fills missing values (NaN) due to holidays or non-trading days with the last known price
    data = data.ffill()

    # Remove columns that still contain NaN
    # Crucial for matrix algebra: we cannot have "holes" in the return matrix.
    # If a stock wasn't listed at 'start_date', it will be dropped here.
    data = data.dropna(axis=1)

    # Verification: Check if any tickers were dropped during cleaning
    downloaded_tickers = data.columns.tolist()
    if len(downloaded_tickers) < len(tickers):
        dropped_count = len(tickers) - len(downloaded_tickers)
        print(f"Warning: {dropped_count} tickers removed due to incomplete history.")
        print(f"Remaining tickers ({len(downloaded_tickers)}): {downloaded_tickers}")

    print("Successfully downloaded and cleaned data!")
    print("========== End ==========")

    return data


def process_data(price_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    print("========== Processing Data ==========")

    # Calculate daily simple returns (percentage change)
    # Matches Proposal Eq 3.1: R_t = (P_t - P_t-1) / P_t-1
    # dropna() removes the first row (t=0) which is always NaN
    r_matrix = price_df.pct_change().dropna()

    # Calculate the Expected Return (Mean) for each stock
    # Matches Proposal Eq 3.2: mu = average(R_t)
    mu_vector = r_matrix.mean()

    print("----- Vector expected return (mu) preview: -----")
    print(mu_vector.head().to_string())

    print("----- Matrix return history preview: -----")
    print(r_matrix.head(2))

    print("========== End ==========")
    return r_matrix, mu_vector
