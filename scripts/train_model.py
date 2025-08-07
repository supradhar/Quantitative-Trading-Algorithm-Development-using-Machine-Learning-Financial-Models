import sys
import os
import argparse
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import logging
import pandas as pd
import numpy as np
from src.data.data_collector import ForexDataCollector
from src.models.ml_models import ForexMLPipeline
from src.utils.helpers import setup_logging, load_config, create_directories

def main():
    parser = argparse.ArgumentParser(description='Train forex trading ML models')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file path')
    parser.add_argument('--symbols', nargs='+', default=None, help='Forex symbols to train on')
    parser.add_argument('--period', type=str, default='2y', help='Data period to fetch')
    parser.add_argument('--output-dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--force-retrain', action='store_true', help='Force retrain even if models exist')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("="*50)
    logger.info("FOREX TRADING ML MODEL TRAINING")
    logger.info("="*50)
    
    try:
        # Create necessary directories
        create_directories()
        
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Configuration loaded from {args.config}")
        
        # Get symbols
        symbols = args.symbols or config.get('data', {}).get('symbols', ['EURUSD', 'GBPUSD', 'USDJPY'])
        logger.info(f"Training on symbols: {symbols}")
        
        # Check if models already exist
        output_dir = os.path.abspath(args.output_dir)
        if os.path.exists(output_dir) and os.listdir(output_dir) and not args.force_retrain:
            response = input("Models already exist. Continue training? (y/N): ")
            if response.lower() != 'y':
                logger.info("Training cancelled by user")
                return
        
        # Initialize data collector
        logger.info("Initializing data collector...")
        data_collector = ForexDataCollector(symbols)
        
        # Collect data
        logger.info(f"Collecting forex data for period: {args.period}")
        data = data_collector.fetch_multiple_symbols(period=args.period, interval="1d")
        
        if data.empty:
            logger.error("No data collected. Please check your internet connection and symbol names.")
            return
        
        logger.info(f"Collected {len(data)} total records")
        logger.info(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
        
        # Data quality checks
        logger.info("Performing data quality checks...")
        for symbol in symbols:
            symbol_data = data[data['Symbol'] == symbol]
            logger.info(f"{symbol}: {len(symbol_data)} records")
            
            if len(symbol_data) < 100:
                logger.warning(f"Warning: {symbol} has only {len(symbol_data)} records")
        
        # Add technical indicators
        logger.info("Calculating technical indicators...")
        all_data_with_indicators = []
        
        for symbol in symbols:
            symbol_data = data[data['Symbol'] == symbol].copy()
            symbol_indicators = data_collector.calculate_technical_indicators(symbol_data)
            all_data_with_indicators.append(symbol_indicators)
        
        data_with_indicators = pd.concat(all_data_with_indicators, ignore_index=True)
        logger.info("Technical indicators added successfully")
        
        # Initialize ML pipeline
        logger.info("Initializing ML pipeline...")
        ml_pipeline = ForexMLPipeline(config)
        
        # Prepare features
        logger.info("Preparing features and targets...")
        X, y_reg, y_clf = ml_pipeline.prepare_features(data_with_indicators)
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Regression targets: {len(y_reg[~np.isnan(y_reg)])}")
        logger.info(f"Classification targets: {len(y_clf[~np.isnan(y_clf)])}")
        
        if X.shape[0] == 0:
            logger.error("No valid features prepared. Check your data.")
            return
        
        # Train models
        logger.info("Starting model training...")
        logger.info("This may take several minutes depending on data size and hyperparameter tuning...")
        
        ml_pipeline.train_models(X, y_reg, y_clf)
        
        # Save models
        logger.info(f"Saving models to {output_dir}/")
        os.makedirs(output_dir, exist_ok=True)
        ml_pipeline.save_models(f"{output_dir}/")
        
        # Save training metadata
        metadata = {
            'training_date': pd.Timestamp.now().isoformat(),
            'symbols': symbols,
            'data_period': args.period,
            'total_records': len(data),
            'feature_count': X.shape[1],
            'models_trained': list(ml_pipeline.models.keys()),
            'config': config
        }
        
        metadata_df = pd.DataFrame([metadata])
        metadata_df.to_csv(f"{output_dir}/training_metadata.csv", index=False)
        
        logger.info("="*50)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*50)
        logger.info(f"Models saved to: {output_dir}/")
        logger.info(f"Available models: {list(ml_pipeline.models.keys())}")
        logger.info("\nNext steps:")
        logger.info("1. Start the API server: uvicorn src.api.app:app --reload")
        logger.info("2. Test predictions: python scripts/predict.py --symbol EURUSD")
        logger.info("3. View API docs: http://localhost:8000/docs")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
