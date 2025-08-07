import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

class ForexPreprocessor:
    """Preprocess forex data for ML models"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.logger = logging.getLogger(__name__)
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate forex data"""
        df_clean = df.copy()
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        # Sort by date
        df_clean = df_clean.sort_values('Date')
        
        # Remove rows with invalid prices
        price_cols = ['Open', 'High', 'Low', 'Close']
        df_clean = df_clean.dropna(subset=price_cols)
        
        # Validate OHLC data
        valid_ohlc = (
            (df_clean['High'] >= df_clean['Low']) &
            (df_clean['High'] >= df_clean['Open']) &
            (df_clean['High'] >= df_clean['Close']) &
            (df_clean['Low'] <= df_clean['Open']) &
            (df_clean['Low'] <= df_clean['Close']) &
            (df_clean['Open'] > 0) &
            (df_clean['Close'] > 0)
        )
        
        df_clean = df_clean[valid_ohlc]
        
        self.logger.info(f"Cleaned data: {len(df_clean)} records remain")
        return df_clean
    
    def create_lag_features(self, df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
        """Create lag features"""
        df_with_lags = df.copy()
        
        for col in columns:
            for lag in lags:
                df_with_lags[f'{col}_lag_{lag}'] = df_with_lags[col].shift(lag)
                
        return df_with_lags
    
    def create_rolling_features(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Create rolling window features"""
        df_rolling = df.copy()
        
        for window in windows:
            df_rolling[f'Close_rolling_mean_{window}'] = df_rolling['Close'].rolling(window).mean()
            df_rolling[f'Close_rolling_std_{window}'] = df_rolling['Close'].rolling(window).std()
            df_rolling[f'Volume_rolling_mean_{window}'] = df_rolling.get('Volume', pd.Series()).rolling(window).mean()
            
        return df_rolling
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features"""
        df_engineered = df.copy()
        
        # Price-based features
        df_engineered['Price_Range'] = df_engineered['High'] - df_engineered['Low']
        df_engineered['Price_Change'] = df_engineered['Close'] - df_engineered['Open']
        df_engineered['Price_Change_Pct'] = df_engineered['Price_Change'] / df_engineered['Open']
        
        # Volatility features
        df_engineered['Intraday_Volatility'] = df_engineered['Price_Range'] / df_engineered['Open']
        
        # Trend features
        df_engineered['Upper_Shadow'] = df_engineered['High'] - np.maximum(df_engineered['Open'], df_engineered['Close'])
        df_engineered['Lower_Shadow'] = np.minimum(df_engineered['Open'], df_engineered['Close']) - df_engineered['Low']
        
        # Time-based features
        if 'Date' in df_engineered.columns:
            df_engineered['Date'] = pd.to_datetime(df_engineered['Date'])
            df_engineered['DayOfWeek'] = df_engineered['Date'].dt.dayofweek
            df_engineered['Month'] = df_engineered['Date'].dt.month
            df_engineered['Quarter'] = df_engineered['Date'].dt.quarter
            df_engineered['IsMonthEnd'] = df_engineered['Date'].dt.is_month_end.astype(int)
            
        return df_engineered
