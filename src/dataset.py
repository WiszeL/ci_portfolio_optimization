import yfinance as yf
import pandas as pd

def download_data(tickers, start, end):
  print('========== Downloading Data ==========')
  print(f'Downloading data for {len(tickers)} stock From {start} to {end}...')

  # Download
  data: pd.DataFrame = yf.download(tickers, start, end, auto_adjust=True)['Close']
  
  # Filling NaN due to suspended or other cause
  # Fill with the previous day
  data = data.ffill()
  data = data.dropna(axis=1)

  # Check is there removed ticker
  downloaded_tickers = data.columns.tolist()
  if len(downloaded_tickers) < len(tickers):
    print('Some removed due to full of NaN')
    print('Downloaded tickers:', downloaded_tickers)

  if not data.empty:
    data = data.iloc[1:]

  print('Successfully download and cleaning data!')
  print('========== End ==========')
  
  return data

def process_data(price_df: pd.DataFrame):
  print('========== Processing Data ==========')
  # Get simple return and expected return
  r_matrix = price_df.pct_change().dropna()
  mu_vector = r_matrix.mean()

  # Display
  print('----- Vector expected return (mu): -----')
  print(mu_vector.to_string())

  print('----- Matrix return history: -----')
  print(r_matrix.head(2))

  # Save to CSV
  print('========== End ==========')
  return r_matrix, mu_vector
