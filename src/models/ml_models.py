import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb
import optuna
from typing import Dict, Tuple, Any
import joblib
import logging

class ForexMLPipeline:
    """Complete ML pipeline for forex prediction"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.logger = logging.getLogger(__name__)
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets"""
        
        # Feature engineering
        feature_cols = [
            'SMA_10', 'SMA_20', 'RSI_14', 'MACD', 'MACD_Signal',
            'BB_Upper', 'BB_Lower', 'Volatility'
        ]
        
        # Create lag features
        for col in ['Close', 'Returns']:
            for lag in [1, 2, 3, 5]:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                feature_cols.append(f'{col}_lag_{lag}')
        
        # Create target variables
        df['Return_1d'] = df['Close'].pct_change().shift(-1)  # Next day return
        df['High_Volatility'] = (df['Volatility'] > df['Volatility'].quantile(0.8)).astype(int)
        
        # Remove NaN values
        df_clean = df.dropna()
        
        X = df_clean[feature_cols].values
        y_regression = df_clean['Return_1d'].values
        y_classification = df_clean['High_Volatility'].values
        
        return X, y_regression, y_classification
    
    def optimize_hyperparameters(self, 
                                X: np.ndarray, 
                                y: np.ndarray, 
                                model_type: str = 'xgboost') -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna"""
        
        def objective(trial):
            if model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'subsample': trial.suggest_float('subsample', 0.8, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1.0),
                }
                model = xgb.XGBRegressor(**params, random_state=42)
                
            elif model_type == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                }
                model = RandomForestRegressor(**params, random_state=42)
            
            scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
            return -scores.mean()
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.config.get('hyperparameter_tuning', {}).get('n_trials', 100))
        
        return study.best_params
    
    def train_models(self, X: np.ndarray, y_reg: np.ndarray, y_clf: np.ndarray):
        """Train multiple models"""
        
        # Split data
        X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
            X, y_reg, y_clf, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['feature_scaler'] = scaler
        
        # Train regression models
        self.logger.info("Training regression models...")
        
        # XGBoost Regressor
        best_params_xgb = self.optimize_hyperparameters(X_train_scaled, y_reg_train, 'xgboost')
        xgb_reg = xgb.XGBRegressor(**best_params_xgb, random_state=42)
        xgb_reg.fit(X_train_scaled, y_reg_train)
        self.models['xgb_regressor'] = xgb_reg
        
        # Random Forest Regressor
        best_params_rf = self.optimize_hyperparameters(X_train_scaled, y_reg_train, 'random_forest')
        rf_reg = RandomForestRegressor(**best_params_rf, random_state=42)
        rf_reg.fit(X_train_scaled, y_reg_train)
        self.models['rf_regressor'] = rf_reg
        
        # Train classification models
        self.logger.info("Training classification models...")
        
        # XGBoost Classifier
        xgb_clf = xgb.XGBClassifier(**best_params_xgb, random_state=42)
        xgb_clf.fit(X_train_scaled, y_clf_train)
        self.models['xgb_classifier'] = xgb_clf
        
        # Random Forest Classifier
        rf_clf = RandomForestClassifier(**best_params_rf, random_state=42)
        rf_clf.fit(X_train_scaled, y_clf_train)
        self.models['rf_classifier'] = rf_clf
        
        # Evaluate models
        self._evaluate_models(X_test_scaled, y_reg_test, y_clf_test)
        
    def _evaluate_models(self, X_test: np.ndarray, y_reg_test: np.ndarray, y_clf_test: np.ndarray):
        """Evaluate trained models"""
        
        self.logger.info("Evaluating models...")
        
        # Regression evaluation
        for name, model in self.models.items():
            if 'regressor' in name:
                pred = model.predict(X_test)
                mse = mean_squared_error(y_reg_test, pred)
                self.logger.info(f"{name} MSE: {mse:.6f}")
                
        # Classification evaluation  
        for name, model in self.models.items():
            if 'classifier' in name:
                pred = model.predict(X_test)
                acc = accuracy_score(y_clf_test, pred)
                self.logger.info(f"{name} Accuracy: {acc:.4f}")
    
    def predict(self, X: np.ndarray, model_name: str = 'xgb_regressor') -> np.ndarray:
        """Make predictions"""
        
        X_scaled = self.scalers['feature_scaler'].transform(X)
        return self.models[model_name].predict(X_scaled)
    
    def save_models(self, path: str = "models/"):
        """Save trained models"""

        import os
        path = os.path.abspath(path)
        for name, model in self.models.items():
            joblib.dump(model, f"{path}/{name}.pkl")

        for name, scaler in self.scalers.items():
            joblib.dump(scaler, f"{path}/{name}.pkl")

        self.logger.info(f"Models saved to {path}")

    def load_models(self, path: str = "models/"):
        """Load trained models"""

        import os
        path = os.path.abspath(path)
        for file in os.listdir(path):
            if file.endswith('.pkl'):
                name = file.replace('.pkl', '')
                if 'scaler' in name:
                    self.scalers[name] = joblib.load(f"{path}/{file}")
                else:
                    self.models[name] = joblib.load(f"{path}/{file}")

        self.logger.info(f"Models loaded from {path}")
