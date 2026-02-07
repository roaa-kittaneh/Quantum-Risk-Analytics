import os
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class DataService:
    """
    Service responsible for data retrieval and initial processing.
    Separates the data layer from business logic.
    """

    @staticmethod
    def load_financial_data() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Loads both Market Returns and Loan Portfolio Data from CSV files."""
        market_df, loan_df = None, None
        
        try:
            # Load Market Data
            if os.path.exists('market_returns.csv'):
                market_df = pd.read_csv('market_returns.csv')
                logger.info(f"Loaded Market Data: {len(market_df)} records")
            
            # Load Loan Data
            if os.path.exists('loan_data.csv'):
                loan_df = pd.read_csv('loan_data.csv')
                logger.info(f"Loaded Loan Data: {len(loan_df)} loans")
            else:
                logger.warning("loan_data.csv not found! Credit risk will be zero in simulation.")
                
            return market_df, loan_df
        except Exception as e:
            logger.error(f"Data loading error: {str(e)}")
            return None, None

    @staticmethod
    def get_mock_market_data() -> pd.DataFrame:
        """Generates mock data to prevent system crash if CSVs are missing."""
        return pd.DataFrame({
            'Gold': np.random.normal(0, 0.01, 100), 
            'SP500': np.random.normal(0, 0.02, 100)
        })

    @staticmethod
    def calculate_portfolio_volatility(market_data: pd.DataFrame, 
                                       volatility_multiplier: float) -> float:
        """
        Calculates the combined volatility of a 50/50 Gold/SP500 portfolio.
        Includes correlation effects.
        """
        # Fallback if columns don't exist
        if 'Gold' not in market_data.columns:
            return 0.02 * volatility_multiplier

        sigma_gold = market_data['Gold'].std()
        sigma_sp500 = market_data['SP500'].std()
        correlation = market_data['Gold'].corr(market_data['SP500'])
        
        # Portfolio weights (50/50)
        w1, w2 = 0.5, 0.5
        
        # Portfolio variance formula: Var(aX + bY) = a^2*Var(X) + b^2*Var(Y) + 2ab*Cov(X,Y)
        portfolio_variance = (
            (w1**2 * sigma_gold**2) +
            (w2**2 * sigma_sp500**2) +
            (2 * w1 * w2 * sigma_gold * sigma_sp500 * correlation)
        )
        
        base_sigma = np.sqrt(portfolio_variance)
        adjusted_sigma = base_sigma * volatility_multiplier
        
        logger.info(f"Portfolio volatility: base={base_sigma:.6f}, adjusted={adjusted_sigma:.6f}")
        
        return adjusted_sigma
