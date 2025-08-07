# ==================== src/__init__.py ====================
# Empty file to make src a package

# ==================== src/data/__init__.py ====================
# Empty file to make data a package

# ==================== src/models/__init__.py ====================
# Empty file to make models a package

# ==================== src/utils/__init__.py ====================
# Empty file to make utils a package

# ==================== src/api/__init__.py ====================
# Empty file to make api a package

# ==================== src/utils/helpers.py ====================
import logging
import yaml
import os
from typing import Dict, Any
import pandas as pd
import numpy as np

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('forex_ml.log')
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        # Return default config if file not found
        return {
            'data': {
                'symbols': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'],
                'timeframe': '1D',
                'lookback_days': 252,
                'train_split': 0.8
            },
            'models': {
                'random_forest': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': 42
                },
                'xgboost': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'random_state': 42
                }
            },
            'hyperparameter_tuning': {
                'n_trials': 50,
                'cv_folds': 5
            }
        }

def create_directories():
    """Create necessary directories"""
    directories = [
        'data/raw',
        'data/processed', 
        'models',
        'logs',
        'notebooks',
        'tests'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)