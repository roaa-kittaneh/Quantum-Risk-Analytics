import logging

# ==================== SCENARIO CONFIGURATION ====================
SCENARIOS = {
    "mild": {
        "name": "Mild Disruption",
        "description": "Correction (0.8x) - Minor market correction",
        "volatility_multiplier": 0.8,
        "shock_multiplier": 0.8,
        "mean_drift": 0.02,
        "risk_level": "LOW",
        "credit_stress_factor": 1.1
    },
    "baseline": {
        "name": "Baseline Crisis",
        "description": "Financial Crisis (1.5x) - Systemic shock",
        "volatility_multiplier": 1.5,
        "shock_multiplier": 1.2,
        "mean_drift": 0.01,
        "risk_level": "MEDIUM",
        "credit_stress_factor": 2.0
    },
    "super": {
        "name": "Future Super-Crisis",
        "description": "Extreme Event (4.0x) - Market Collapse",
        "volatility_multiplier": 4.0,
        "shock_multiplier": 3.0,
        "mean_drift": -0.05,
        "risk_level": "EXTREME",
        "credit_stress_factor": 5.0
    }
}

# ==================== LOGGING CONFIGURATION ====================
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
