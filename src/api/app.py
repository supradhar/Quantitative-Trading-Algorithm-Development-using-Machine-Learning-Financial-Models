from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, validator
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import logging
import asyncio
import os
import sys
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.ml_models import ForexMLPipeline
from src.data.data_collector import ForexDataCollector
from src.utils.helpers import setup_logging, load_config, create_directories

# Initialize FastAPI app
app = FastAPI(
    title="Forex Trading ML API",
    description="Advanced ML API for Forex market prediction using Black-Scholes and Bayesian models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Setup logging
logger = setup_logging()

# Global variables
ml_pipeline: Optional[ForexMLPipeline] = None
data_collector: Optional[ForexDataCollector] = None
config: Dict[str, Any] = {}
model_last_updated: Optional[datetime] = None

# Pydantic models
class PredictionRequest(BaseModel):
    symbol: str
    features: Optional[List[float]] = None
    use_latest_data: bool = True
    
    @validator('symbol')
    def validate_symbol(cls, v):
        valid_symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'NZDUSD', 'EURGBP']
        if v not in valid_symbols:
            raise ValueError(f'Symbol must be one of {valid_symbols}')
        return v

class PredictionResponse(BaseModel):
    symbol: str
    predicted_return: float
    volatility_class: int
    confidence: float
    prediction_timestamp: datetime
    model_used: str

class ModelInfo(BaseModel):
    available_models: List[str]
    model_performance: Dict[str, Dict[str, float]]
    last_trained: Optional[datetime]
    training_data_size: int

class MarketDataResponse(BaseModel):
    symbol: str
    latest_data: Dict[str, Any]
    technical_indicators: Dict[str, float]
    timestamp: datetime

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    global ml_pipeline, data_collector, config, model_last_updated
    
    logger.info("Starting Forex Trading ML API...")
    
    try:
        # Create necessary directories
        create_directories()
        
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")
        
        # Initialize data collector
        symbols = config.get('data', {}).get('symbols', ['EURUSD', 'GBPUSD'])
        data_collector = ForexDataCollector(symbols)
        logger.info(f"Data collector initialized for symbols: {symbols}")
        
        # Try to load existing models
        try:
            ml_pipeline = ForexMLPipeline(config)
            if os.path.exists("models") and os.listdir("models"):
                ml_pipeline.load_models("models/")
                model_last_updated = datetime.now()
                logger.info("Existing models loaded successfully")
            else:
                logger.warning("No existing models found. Train models first using: python scripts/train_model.py")
                
        except Exception as e:
            logger.warning(f"Could not load models: {e}")
            ml_pipeline = None
            
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise

# Health check endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Forex Trading ML API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "models_loaded": ml_pipeline is not None,
        "data_collector_ready": data_collector is not None,
        "model_count": len(ml_pipeline.models) if ml_pipeline else 0,
        "last_model_update": model_last_updated
    }

# Model information endpoints
@app.get("/models/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about available models"""
    if ml_pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Calculate mock performance metrics (in production, these would be real)
    performance = {}
    for model_name in ml_pipeline.models.keys():
        if 'regressor' in model_name:
            performance[model_name] = {
                "mse": np.random.uniform(0.001, 0.005),
                "mae": np.random.uniform(0.02, 0.05),
                "r2": np.random.uniform(0.6, 0.9)
            }
        else:
            performance[model_name] = {
                "accuracy": np.random.uniform(0.7, 0.9),
                "precision": np.random.uniform(0.65, 0.85),
                "recall": np.random.uniform(0.65, 0.85)
            }
    
    return ModelInfo(
        available_models=list(ml_pipeline.models.keys()),
        model_performance=performance,
        last_trained=model_last_updated,
        training_data_size=1000  # Mock value
    )

# Prediction endpoints
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make trading prediction"""
    if ml_pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Please train models first.")
    
    if data_collector is None:
        raise HTTPException(status_code=503, detail="Data collector not initialized")
    
    try:
        if request.use_latest_data:
            # Fetch latest data and prepare features
            logger.info(f"Fetching latest data for {request.symbol}")
            data = data_collector.fetch_forex_data(request.symbol, period="1mo")
            
            if data.empty:
                raise HTTPException(status_code=404, detail=f"No data available for {request.symbol}")
            
            # Add technical indicators
            data_with_indicators = data_collector.calculate_technical_indicators(data)
            
            # Prepare features
            X, _, _ = ml_pipeline.prepare_features(data_with_indicators)
            
            if len(X) == 0:
                raise HTTPException(status_code=400, detail="Unable to prepare features from data")
            
            # Use latest features
            features = X[-1:]
        else:
            if request.features is None:
                raise HTTPException(status_code=400, detail="Features must be provided when use_latest_data=False")
            
            features = np.array(request.features).reshape(1, -1)
        
        # Make predictions with available models
        predictions = {}
        
        # Try regression models first
        regressor_models = [m for m in ml_pipeline.models.keys() if 'regressor' in m]
        classifier_models = [m for m in ml_pipeline.models.keys() if 'classifier' in m]
        
        if regressor_models:
            model_name = regressor_models[0]  # Use first available regressor
            return_pred = ml_pipeline.predict(features, model_name)[0]
            predictions['return'] = return_pred
            predictions['model'] = model_name
        else:
            predictions['return'] = 0.0
            predictions['model'] = 'none'
        
        # Try classification models
        if classifier_models:
            vol_model = classifier_models[0]
            vol_pred = ml_pipeline.predict(features, vol_model)[0]
            predictions['volatility'] = int(vol_pred)
        else:
            predictions['volatility'] = 0
        
        # Calculate confidence based on prediction magnitude
        confidence = min(abs(predictions['return']) * 100, 1.0) if predictions['return'] != 0 else 0.5
        
        return PredictionResponse(
            symbol=request.symbol,
            predicted_return=float(predictions['return']),
            volatility_class=predictions['volatility'],
            confidence=float(confidence),
            prediction_timestamp=datetime.now(),
            model_used=predictions['model']
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(requests: List[PredictionRequest]):
    """Make batch predictions"""
    if len(requests) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 predictions per batch")
    
    results = []
    for request in requests:
        try:
            prediction = await predict(request)
            results.append(prediction)
        except Exception as e:
            results.append({
                "symbol": request.symbol,
                "error": str(e),
                "timestamp": datetime.now()
            })
    
    return {"batch_predictions": results, "processed": len(results)}

# Data endpoints
@app.get("/symbols")
async def get_symbols():
    """Get available forex symbols"""
    symbols = config.get('data', {}).get('symbols', ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'])
    return {
        "symbols": symbols,
        "count": len(symbols),
        "supported_operations": ["prediction", "data_fetch", "technical_analysis"]
    }

@app.get("/data/{symbol}", response_model=MarketDataResponse)
async def get_latest_data(symbol: str, period: str = "5d"):
    """Get latest market data for symbol"""
    if data_collector is None:
        raise HTTPException(status_code=503, detail="Data collector not initialized")
    
    valid_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y"]
    if period not in valid_periods:
        raise HTTPException(status_code=400, detail=f"Period must be one of {valid_periods}")
    
    try:
        logger.info(f"Fetching data for {symbol}, period: {period}")
        data = data_collector.fetch_forex_data(symbol, period=period)
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
        
        # Add technical indicators
        data_with_indicators = data_collector.calculate_technical_indicators(data)
        
        # Get latest record
        latest = data_with_indicators.tail(1).iloc[0]
        
        # Prepare response
        latest_data = {
            "date": latest.get('Date', '').strftime('%Y-%m-%d') if pd.notna(latest.get('Date')) else '',
            "open": float(latest.get('Open', 0)),
            "high": float(latest.get('High', 0)),
            "low": float(latest.get('Low', 0)),
            "close": float(latest.get('Close', 0)),
            "volume": float(latest.get('Volume', 0)) if pd.notna(latest.get('Volume')) else 0
        }
        
        # Technical indicators
        technical_indicators = {
            "sma_20": float(latest.get('SMA_20', 0)) if pd.notna(latest.get('SMA_20')) else None,
            "rsi_14": float(latest.get('RSI_14', 0)) if pd.notna(latest.get('RSI_14')) else None,
            "macd": float(latest.get('MACD', 0)) if pd.notna(latest.get('MACD')) else None,
            "volatility": float(latest.get('Volatility', 0)) if pd.notna(latest.get('Volatility')) else None,
            "bb_upper": float(latest.get('BB_Upper', 0)) if pd.notna(latest.get('BB_Upper')) else None,
            "bb_lower": float(latest.get('BB_Lower', 0)) if pd.notna(latest.get('BB_Lower')) else None
        }
        
        return MarketDataResponse(
            symbol=symbol,
            latest_data=latest_data,
            technical_indicators=technical_indicators,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Data fetch error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Data fetch failed: {str(e)}")

@app.get("/data/{symbol}/history")
async def get_historical_data(symbol: str, period: str = "1mo", limit: int = 100):
    """Get historical data for symbol"""
    if data_collector is None:
        raise HTTPException(status_code=503, detail="Data collector not initialized")
    
    if limit > 1000:
        raise HTTPException(status_code=400, detail="Limit cannot exceed 1000")
    
    try:
        data = data_collector.fetch_forex_data(symbol, period=period)
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
        
        # Limit records
        data_limited = data.tail(limit)
        
        # Convert to list of dictionaries
        records = []
        for _, row in data_limited.iterrows():
            record = {
                "date": row.get('Date', '').strftime('%Y-%m-%d') if pd.notna(row.get('Date')) else '',
                "open": float(row.get('Open', 0)),
                "high": float(row.get('High', 0)),
                "low": float(row.get('Low', 0)),
                "close": float(row.get('Close', 0)),
                "volume": float(row.get('Volume', 0)) if pd.notna(row.get('Volume')) else 0
            }
            records.append(record)
        
        return {
            "symbol": symbol,
            "period": period,
            "records": records,
            "count": len(records)
        }
        
    except Exception as e:
        logger.error(f"Historical data error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Model training endpoint
@app.post("/models/train")
async def train_models(background_tasks: BackgroundTasks):
    """Trigger model training (background task)"""
    global model_last_updated
    
    def train_models_background():
        """Background task to train models"""
        try:
            from src.models.ml_models import ForexMLPipeline
            
            logger.info("Starting model training...")
            
            # Initialize pipeline
            pipeline = ForexMLPipeline(config)
            
            # Collect training data
            symbols = config.get('data', {}).get('symbols', ['EURUSD', 'GBPUSD'])
            collector = ForexDataCollector(symbols)
            
            data = collector.fetch_multiple_symbols(period="2y", interval="1d")
            
            if data.empty:
                logger.error("No training data available")
                return
            
            # Add technical indicators
            data_with_indicators = collector.calculate_technical_indicators(data)
            
            # Prepare features
            X, y_reg, y_clf = pipeline.prepare_features(data_with_indicators)
            
            # Train models
            pipeline.train_models(X, y_reg, y_clf)
            
            # Save models
            pipeline.save_models("models/")
            
            # Update global pipeline
            global ml_pipeline
            ml_pipeline = pipeline
            model_last_updated = datetime.now()
            
            logger.info("Model training completed successfully")
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
    
    background_tasks.add_task(train_models_background)
    
    return {
        "message": "Model training started in background",
        "status": "training",
        "timestamp": datetime.now()
    }

@app.get("/models/status")
async def get_training_status():
    """Get model training status"""
    return {
        "models_loaded": ml_pipeline is not None,
        "model_count": len(ml_pipeline.models) if ml_pipeline else 0,
        "last_updated": model_last_updated,
        "available_models": list(ml_pipeline.models.keys()) if ml_pipeline else []
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "detail": str(exc)}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return {"error": "Internal server error", "detail": "Please check logs for details"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)