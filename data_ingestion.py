import yfinance as yf
import pandas as pd

def get_market_data(ticker_symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical market data for a given ticker.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period=period, interval=interval)
        return hist
    except Exception as e:
        print(f"Error fetching data for {ticker_symbol}: {e}")
        return pd.DataFrame()

def get_asset_info(ticker_symbol: str) -> dict:
    """
    Fetch fundamental info for a given ticker.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        # Extract only relevant information to keep it concise
        keys_to_keep = [
            "shortName", "symbol", "sector", "industry", "previousClose", 
            "open", "dayLow", "dayHigh", "regularMarketPreviousClose", 
            "marketCap", "volume", "averageVolume", "fiftyTwoWeekLow", 
            "fiftyTwoWeekHigh", "trailingPE", "forwardPE", "dividendYield",
            "beta", "trailingPegRatio", "shortRatio", "fiftyDayAverage", 
            "twoHundredDayAverage", "priceToBook", "debtToEquity", "returnOnEquity",
            "freeCashflow", "operatingMargins", "revenueGrowth", "earningsGrowth"
        ]
        return {k: info.get(k, "N/A") for k in keys_to_keep}
    except Exception as e:
        print(f"Error fetching info for {ticker_symbol}: {e}")
        return {}
