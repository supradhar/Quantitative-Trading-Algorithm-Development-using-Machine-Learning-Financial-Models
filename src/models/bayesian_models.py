import pymc as pm
import numpy as np
import pandas as pd
from typing import Dict, Any
import arviz as az

class BayesianForexModel:
    """Bayesian models for forex prediction"""
    
    def __init__(self):
        self.model = None
        self.trace = None
        
    def build_volatility_model(self, returns: np.ndarray) -> pm.Model:
        """Build Bayesian volatility model (GARCH-like)"""
        
        with pm.Model() as model:
            # Priors
            mu = pm.Normal('mu', mu=0, sigma=0.01)
            alpha0 = pm.Exponential('alpha0', 1)
            alpha1 = pm.Beta('alpha1', alpha=1, beta=1)
            beta1 = pm.Beta('beta1', alpha=1, beta=1)
            
            # Volatility process
            h = pm.Deterministic('h', alpha0 + alpha1 * returns[:-1]**2 + beta1 * alpha0)
            
            # Likelihood
            y = pm.Normal('y', mu=mu, sigma=pm.math.sqrt(h), observed=returns[1:])
            
        self.model = model
        return model
    
    def build_return_prediction_model(self, 
                                    features: np.ndarray, 
                                    returns: np.ndarray) -> pm.Model:
        """Build Bayesian linear regression for return prediction"""
        
        with pm.Model() as model:
            # Priors
            alpha = pm.Normal('alpha', mu=0, sigma=1)
            beta = pm.Normal('beta', mu=0, sigma=1, shape=features.shape[1])
            sigma = pm.HalfNormal('sigma', sigma=1)
            
            # Linear combination
            mu = alpha + pm.math.dot(features, beta)
            
            # Likelihood
            y = pm.Normal('y', mu=mu, sigma=sigma, observed=returns)
            
        self.model = model
        return model
    
    def fit(self, n_samples: int = 2000, chains: int = 4) -> az.InferenceData:
        """Fit the Bayesian model"""
        
        with self.model:
            self.trace = pm.sample(n_samples, chains=chains, return_inferencedata=True)
            
        return self.trace
    
    def predict(self, X_new: np.ndarray, n_samples: int = 1000) -> Dict[str, np.ndarray]:
        """Make predictions with uncertainty quantification"""
        
        with self.model:
            pm.set_data({"X": X_new})
            posterior_pred = pm.sample_posterior_predictive(
                self.trace, samples=n_samples, return_inferencedata=True
            )
            
        predictions = posterior_pred.posterior_predictive['y'].values
        
        return {
            'mean': np.mean(predictions, axis=(0, 1)),
            'std': np.std(predictions, axis=(0, 1)),
            'quantiles': np.percentile(predictions, [2.5, 25, 50, 75, 97.5], axis=(0, 1))
        }