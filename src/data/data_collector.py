import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging
from datetime import datetime, timedelta

class ForexDataCollector:
    """Collect and manage forex data from various sources"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.logger = logging.getLogger(__name__)
        
    def fetch_forex_data(self, 
                        symbol: str, 
                        period: str = "2y",
                        interval: str = "1d") -> pd.DataFrame:
        """Fetch forex data using yfinance"""
        try:
            # Convert forex symbol to Yahoo Finance format
            yahoo_symbol = f"{symbol}=X"
            ticker = yf.Ticker(yahoo_symbol)
            
            data = ticker.history(period=period, interval=interval)
            data.reset_index(inplace=True)
            data['Symbol'] = symbol
            
            self.logger.info(f"Fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_multiple_symbols(self, 
                             period: str = "2y",
                             interval: str = "1d") -> pd.DataFrame:
        """Fetch data for multiple forex pairs"""
        all_data = []
        
        for symbol in self.symbols:
            data = self.fetch_forex_data(symbol, period, interval)
            if not data.empty:
                all_data.append(data)
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            return combined_data
        return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        df_copy = df.copy()
        
        # Simple Moving Averages
        df_copy['SMA_10'] = df_copy['Close'].rolling(window=10).mean()
        df_copy['SMA_20'] = df_copy['Close'].rolling(window=20).mean()
        df_copy['SMA_50'] = df_copy['Close'].rolling(window=50).mean()
        
        # RSI
        delta = df_copy['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_copy['RSI_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df_copy['Close'].ewm(span=12).mean()
        exp2 = df_copy['Close'].ewm(span=26).mean()
        df_copy['MACD'] = exp1 - exp2
        df_copy['MACD_Signal'] = df_copy['MACD'].ewm(span=9).mean()
        
        # Bollinger Bands
        df_copy['BB_Middle'] = df_copy['Close'].rolling(window=20).mean()
        bb_std = df_copy['Close'].rolling(window=20).std()
        df_copy['BB_Upper'] = df_copy['BB_Middle'] + (bb_std * 2)
        df_copy['BB_Lower'] = df_copy['BB_Middle'] - (bb_std * 2)
        
        # Volatility
        df_copy['Returns'] = df_copy['Close'].pct_change()
        df_copy['Volatility'] = df_copy['Returns'].rolling(window=20).std()
        
        return df_copy
