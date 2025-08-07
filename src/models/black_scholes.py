import numpy as np
from scipy.stats import norm
from typing import Tuple, Dict

class BlackScholesModel:
    """Black-Scholes model for options pricing and volatility estimation"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
    
    def calculate_option_price(self, 
                             S: float,  # Current price 
                             K: float,  # Strike price
                             T: float,  # Time to expiration
                             sigma: float,  # Volatility
                             option_type: str = 'call') -> float:
        """Calculate Black-Scholes option price"""
        
        d1 = (np.log(S/K) + (self.risk_free_rate + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type.lower() == 'call':
            price = S*norm.cdf(d1) - K*np.exp(-self.risk_free_rate*T)*norm.cdf(d2)
        else:  # put
            price = K*np.exp(-self.risk_free_rate*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
            
        return price
    
    def calculate_greeks(self, 
                        S: float, K: float, T: float, sigma: float) -> Dict[str, float]:
        """Calculate option Greeks"""
        
        d1 = (np.log(S/K) + (self.risk_free_rate + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        greeks = {
            'delta': norm.cdf(d1),
            'gamma': norm.pdf(d1) / (S * sigma * np.sqrt(T)),
            'theta': -(S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                      self.risk_free_rate * K * np.exp(-self.risk_free_rate * T) * norm.cdf(d2)),
            'vega': S * norm.pdf(d1) * np.sqrt(T),
            'rho': K * T * np.exp(-self.risk_free_rate * T) * norm.cdf(d2)
        }
        
        return greeks
    
    def implied_volatility(self, 
                          market_price: float,
                          S: float, K: float, T: float,
                          option_type: str = 'call',
                          max_iterations: int = 100,
                          tolerance: float = 1e-6) -> float:
        """Calculate implied volatility using Newton-Raphson method"""
        
        sigma = 0.2  # Initial guess
        
        for i in range(max_iterations):
            price = self.calculate_option_price(S, K, T, sigma, option_type)
            vega = self.calculate_greeks(S, K, T, sigma)['vega']
            
            price_diff = price - market_price
            
            if abs(price_diff) < tolerance:
                return sigma
                
            if vega != 0:
                sigma = sigma - price_diff / vega
            else:
                break
                
        return sigma
