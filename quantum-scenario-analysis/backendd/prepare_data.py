"""
Professional Market & Credit Risk Data Generator
Realistic financial time series with integrated loan portfolio
- Market data: Correlated asset returns with GARCH effects
- Loan data: Credit exposure correlated with market stress
- Systemic linkage: Default rates spike during market crises
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RealisticFinancialDataGenerator:
    """
    Professional financial data generator with:
    1. Market data (Gold, SP500) with proper correlation
    2. Loan portfolio with market-correlated default risk
    3. Systemic linkage between market stress and credit losses
    """
    
    # Market parameters (empirically calibrated)
    MARKET_PARAMS = {
        'sp500': {
            'annual_return': 0.08,
            'annual_volatility': 0.18,
            'crisis_vol_multiplier': 2.5
        },
        'gold': {
            'annual_return': 0.02,
            'annual_volatility': 0.12,
            'crisis_vol_multiplier': 1.5
        },
        'correlation': -0.3  # Negative correlation (safe haven)
    }
    
    # Credit risk parameters (industry benchmarks)
    CREDIT_PARAMS = {
        'base_default_rate': 0.03,      # 3% baseline default (normal times)
        'crisis_default_rate': 0.12,    # 12% default in severe crisis
        'market_sensitivity': 0.5,      # How much market affects credit
        'loan_amount_mean': 50000,      # Average loan size
        'loan_amount_std': 25000,       # Loan size variation
        'recovery_rate': 0.40           # 40% recovery on defaulted loans
    }
    
    # Historical crisis periods
    CRISIS_PERIODS = [
        {'name': '2008 Financial Crisis', 'start': '2008-09-15', 'end': '2009-03-09', 'severity': 3.0},
        {'name': 'COVID-19 Crash', 'start': '2020-02-20', 'end': '2020-03-23', 'severity': 4.0},
        {'name': 'Dot-com Crash', 'start': '2000-03-10', 'end': '2002-10-09', 'severity': 2.0},
        {'name': 'European Debt Crisis', 'start': '2011-08-01', 'end': '2011-10-04', 'severity': 1.5},
        {'name': '2022 Bear Market', 'start': '2022-01-03', 'end': '2022-10-13', 'severity': 1.8}
    ]
    
    def __init__(self, start_date: str = "1990-01-01", end_date: str = "2024-12-31"):
        self.dates = pd.date_range(start=start_date, end=end_date, freq='B')
        self.n_days = len(self.dates)
        self.daily_params = self._annualize_to_daily(self.MARKET_PARAMS)
        
        # Initialize random seed for reproducibility
        np.random.seed(42)
        
        logger.info(f"Initialized generator: {len(self.dates)} trading days")
    
    def _annualize_to_daily(self, params: Dict) -> Dict:
        """Convert annual parameters to daily"""
        daily = {}
        for asset, config in params.items():
            if asset == 'correlation':
                daily['correlation'] = config
            else:
                daily[asset] = {
                    'daily_return': config['annual_return'] / 252,
                    'daily_volatility': config['annual_volatility'] / np.sqrt(252),
                    'crisis_multiplier': config['crisis_vol_multiplier']
                }
        return daily
    
    def _generate_correlated_returns(self, correlation: float, n_samples: int, 
                                     df: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Generate correlated returns using Cholesky decomposition"""
        corr_matrix = np.array([
            [1.0, correlation],
            [correlation, 1.0]
        ])
        
        L = np.linalg.cholesky(corr_matrix)
        independent = np.random.standard_t(df=df, size=(n_samples, 2))
        correlated = independent @ L.T
        
        return correlated[:, 0], correlated[:, 1]
    
    def _identify_crisis_periods(self) -> np.ndarray:
        """Identify crisis periods and return severity multipliers"""
        crisis_multipliers = np.ones(self.n_days)
        
        for crisis in self.CRISIS_PERIODS:
            crisis_start = pd.Timestamp(crisis['start'])
            crisis_end = pd.Timestamp(crisis['end'])
            mask = (self.dates >= crisis_start) & (self.dates <= crisis_end)
            crisis_multipliers[mask] = crisis['severity']
            
            logger.info(f"Crisis: {crisis['name']} ({sum(mask)} days, {crisis['severity']}x)")
        
        return crisis_multipliers
    
    def _apply_garch_volatility(self, returns: np.ndarray, base_vol: float, 
                               persistence: float = 0.85) -> np.ndarray:
        """Apply GARCH volatility clustering"""
        n = len(returns)
        volatility = np.zeros(n)
        volatility[0] = base_vol
        
        for t in range(1, n):
            shock = abs(returns[t-1])
            volatility[t] = persistence * volatility[t-1] + (1 - persistence) * shock
        
        volatility = np.maximum(volatility, base_vol * 0.5)
        return volatility
    
    def generate_market_data(self, output_file: str = 'market_returns.csv') -> pd.DataFrame:
        """
        Generate realistic market returns with:
        - Proper correlation
        - Fat tails  
        - Volatility clustering (GARCH)
        - Historical crisis periods
        """
        logger.info("Generating realistic correlated market returns...")
        
        # Generate correlated innovations
        sp500_innovations, gold_innovations = self._generate_correlated_returns(
            correlation=self.daily_params['correlation'],
            n_samples=self.n_days,
            df=5
        )
        
        actual_corr = np.corrcoef(sp500_innovations, gold_innovations)[0, 1]
        logger.info(f"Generated correlation: {actual_corr:.3f}")
        
        # Identify crisis periods
        crisis_multipliers = self._identify_crisis_periods()
        
        # Get parameters
        sp500_params = self.daily_params['sp500']
        gold_params = self.daily_params['gold']
        
        # Initialize returns
        sp500_returns = np.zeros(self.n_days)
        gold_returns = np.zeros(self.n_days)
        
        # Generate returns with crisis-adjusted volatility
        for t in range(self.n_days):
            sp500_vol = sp500_params['daily_volatility'] * (1 + (crisis_multipliers[t] - 1) * 0.5)
            gold_vol = gold_params['daily_volatility'] * (1 + (crisis_multipliers[t] - 1) * 0.3)
            
            sp500_returns[t] = sp500_params['daily_return'] + sp500_vol * sp500_innovations[t]
            gold_returns[t] = gold_params['daily_return'] + gold_vol * gold_innovations[t]
        
        # Apply GARCH volatility clustering
        sp500_vol_series = self._apply_garch_volatility(sp500_returns, sp500_params['daily_volatility'])
        sp500_returns = sp500_returns * (sp500_vol_series / sp500_params['daily_volatility'])
        
        # Create DataFrame
        returns_df = pd.DataFrame({
            'Gold': gold_returns,
            'SP500': sp500_returns
        }, index=self.dates)
        
        # Save
        returns_df.to_csv(output_file)
        logger.info(f"✓ Market data saved to {output_file}")
        
        # Validation
        self._validate_market_data(returns_df)
        
        return returns_df
    
    def generate_loan_portfolio(self, market_returns: pd.DataFrame, 
                                n_loans: int = 100000,
                                output_file: str = 'loan_data.csv') -> pd.DataFrame:
        """
        Generate loan portfolio with market-correlated default risk
        
        Key Innovation: Default probability linked to market stress
        - During market crashes → higher defaults (systemic risk)
        - During bull markets → lower defaults
        - This captures the AGGREGATION risk that quantum computing analyzes
        
        Args:
            market_returns: DataFrame with market data
            n_loans: Number of loans in portfolio
            output_file: Output filename
            
        Returns:
            DataFrame with loan portfolio
        """
        logger.info(f"Generating {n_loans:,} loan portfolio with market linkage...")
        
        # Calculate market stress index (lower = more stress)
        sp500_cumulative = (1 + market_returns['SP500']).cumprod()
        sp500_drawdown = (sp500_cumulative - sp500_cumulative.cummax()) / sp500_cumulative.cummax()
        
        # Market stress: 0 (normal) to 1 (extreme crisis)
        market_stress = -sp500_drawdown  # Higher when market down
        market_stress = (market_stress - market_stress.min()) / (market_stress.max() - market_stress.min())
        
        # Assign each loan to a random origination date
        loan_origination_indices = np.random.randint(0, len(market_returns), n_loans)
        
        # Loan characteristics
        loan_amounts = np.abs(np.random.normal(
            self.CREDIT_PARAMS['loan_amount_mean'],
            self.CREDIT_PARAMS['loan_amount_std'],
            n_loans
        ))
        
        # CRITICAL: Default probability correlated with market stress
        base_default_prob = self.CREDIT_PARAMS['base_default_rate']
        crisis_default_prob = self.CREDIT_PARAMS['crisis_default_rate']
        sensitivity = self.CREDIT_PARAMS['market_sensitivity']
        
        # Get market stress at origination
        loan_market_stress = market_stress.values[loan_origination_indices]
        
        # Default probability = base + (crisis - base) * stress * sensitivity
        default_probabilities = (
            base_default_prob + 
            (crisis_default_prob - base_default_prob) * loan_market_stress * sensitivity
        )
        
        # Simulate defaults
        defaults = np.random.random(n_loans) < default_probabilities
        
        # Calculate losses (accounting for recovery)
        recovery_rate = self.CREDIT_PARAMS['recovery_rate']
        losses = defaults * loan_amounts * (1 - recovery_rate)
        
        # Create loan DataFrame
        loan_data = pd.DataFrame({
            'LoanID': range(1, n_loans + 1),
            'Amount': loan_amounts,
            'OriginationDate': self.dates[loan_origination_indices],
            'DefaultProbability': default_probabilities,
            'Default': defaults.astype(int),
            'Loss': losses,
            'MarketStress': loan_market_stress
        })
        
        # Statistics
        total_exposure = loan_amounts.sum()
        total_defaults = defaults.sum()
        default_rate = defaults.mean()
        total_loss = losses.sum()
        loss_rate = total_loss / total_exposure
        
        logger.info(f"\n{'='*60}")
        logger.info("LOAN PORTFOLIO STATISTICS")
        logger.info(f"{'='*60}")
        logger.info(f"Total Loans:        {n_loans:,}")
        logger.info(f"Total Exposure:     ${total_exposure:,.0f}")
        logger.info(f"Default Count:      {total_defaults:,} loans")
        logger.info(f"Default Rate:       {default_rate*100:.2f}%")
        logger.info(f"Total Loss:         ${total_loss:,.0f}")
        logger.info(f"Loss Rate:          {loss_rate*100:.2f}%")
        logger.info(f"Avg Loan Size:      ${loan_amounts.mean():,.0f}")
        logger.info(f"Recovery Rate:      {recovery_rate*100:.0f}%")
        
        # Market correlation analysis
        high_stress_mask = loan_market_stress > 0.7
        low_stress_mask = loan_market_stress < 0.3
        
        high_stress_default = defaults[high_stress_mask].mean() if high_stress_mask.any() else 0
        low_stress_default = defaults[low_stress_mask].mean() if low_stress_mask.any() else 0
        
        logger.info(f"\nMARKET CORRELATION:")
        logger.info(f"Default rate (low stress):  {low_stress_default*100:.2f}%")
        logger.info(f"Default rate (high stress): {high_stress_default*100:.2f}%")
        logger.info(f"Stress multiplier:          {high_stress_default/max(low_stress_default, 0.01):.1f}x")
        logger.info(f"{'='*60}\n")
        
        # Save
        loan_data.to_csv(output_file, index=False)
        logger.info(f"✓ Loan portfolio saved to {output_file}")
        
        return loan_data
    
    def _validate_market_data(self, df: pd.DataFrame):
        """Validate market data against empirical benchmarks"""
        logger.info(f"\n{'='*60}")
        logger.info("MARKET DATA VALIDATION")
        logger.info(f"{'='*60}")
        
        for col in df.columns:
            returns = df[col].values
            mean_return = returns.mean() * 252 * 100
            volatility = returns.std() * np.sqrt(252) * 100
            skewness = pd.Series(returns).skew()
            kurtosis = pd.Series(returns).kurtosis()
            
            cumulative = (1 + returns).cumprod()
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_dd = drawdown.min() * 100
            
            logger.info(f"\n{col}:")
            logger.info(f"  Annual Return:  {mean_return:>7.2f}%")
            logger.info(f"  Annual Vol:     {volatility:>7.2f}%")
            logger.info(f"  Skewness:       {skewness:>7.3f}")
            logger.info(f"  Kurtosis:       {kurtosis:>7.3f}")
            logger.info(f"  Max Drawdown:   {max_dd:>7.2f}%")
        
        correlation = df.corr().iloc[0, 1]
        logger.info(f"\nCorrelation: {correlation:.3f}")
        logger.info(f"{'='*60}\n")


def generate_complete_crisis_data():
    """
    Main function to generate complete dataset:
    1. Market returns (Gold, SP500) with correlation and GARCH
    2. Loan portfolio with market-correlated defaults
    """
    logger.info("="*70)
    logger.info("PROFESSIONAL FINANCIAL DATA GENERATION")
    logger.info("Market Data + Credit Portfolio with Systemic Linkage")
    logger.info("="*70 + "\n")
    
    # Initialize generator
    generator = RealisticFinancialDataGenerator(
        start_date="1990-01-01",
        end_date="2024-12-31"
    )
    
    # Generate market data
    market_returns = generator.generate_market_data()
    
    # Generate loan portfolio (LINKED to market stress)
    loan_portfolio = generator.generate_loan_portfolio(
        market_returns=market_returns,
        n_loans=100000
    )
    
    logger.info("\n" + "="*70)
    logger.info("✓ DATA GENERATION COMPLETE")
    logger.info("="*70)
    logger.info("\nGenerated Files:")
    logger.info("  1. market_returns.csv  - Correlated asset returns with GARCH")
    logger.info("  2. loan_data.csv       - Credit portfolio with market linkage")
    logger.info("\nKey Features:")
    logger.info("  ✓ Realistic correlation (Gold vs SP500)")
    logger.info("  ✓ Volatility clustering (GARCH effects)")
    logger.info("  ✓ Historical crisis periods")
    logger.info("  ✓ Market-correlated credit risk")
    logger.info("  ✓ Systemic risk aggregation")
    logger.info("\nReady for Quantum Risk Aggregation Analysis!")
    logger.info("="*70 + "\n")
    
    return market_returns, loan_portfolio


if __name__ == "__main__":
    generate_complete_crisis_data()