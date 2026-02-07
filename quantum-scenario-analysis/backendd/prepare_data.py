import pandas as pd
import numpy as np
import os
import logging

# إعداد نظام التسجيل (Logging) بدلاً من print العادية
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class DataGenerator:
    """Class to handle the generation of crisis-scale financial data."""
    
    def __init__(self, start_date: str = "1990-01-01", end_date: str = "2024-01-01"):
        self.dates = pd.date_range(start=start_date, end=end_date, freq='B')
        self.n_days = len(self.dates)
        np.random.seed(42)

    def generate_market_data(self, output_file: str = 'market_returns.csv'):
        """Generates market returns with Student-t distribution and shocks."""
        logger.info(f"Processing {self.n_days} days of historical market records...")
        
        # Student-t distribution for Fat Tails
        gold_returns = np.random.standard_t(df=3, size=self.n_days) * 0.01 
        sp500_returns = np.random.standard_t(df=3, size=self.n_days) * 0.012
        
        # Global Event Shocks
        for _ in range(5):
            shock_index = np.random.randint(0, self.n_days)
            gold_returns[shock_index] -= 0.05
            sp500_returns[shock_index] -= 0.08
        
        returns_df = pd.DataFrame({'Gold': gold_returns, 'SP500': sp500_returns}, index=self.dates)
        returns_df.to_csv(output_file)
        logger.info(f"SUCCESS: Market data saved to {output_file}")

    def generate_loan_data(self, n_loans: int = 100000, output_file: str = 'loan_data.csv'):
        """Generates large-scale credit portfolio data."""
        logger.info(f"Generating large-scale credit portfolio: {n_loans} records...")
        
        loan_data = pd.DataFrame({
            'LoanID': range(n_loans),
            'Default': np.random.choice([0, 1], size=n_loans, p=[0.92, 0.08]),
            'Amount': np.random.normal(15000, 5000, n_loans)
        })
        loan_data.to_csv(output_file, index=False)
        logger.info(f"SUCCESS: Loan records saved to {output_file}")

if __name__ == "__main__":
    generator = DataGenerator()
    generator.generate_market_data()
    generator.generate_loan_data()
    logger.info("--- Data Preparation Complete! ---")
