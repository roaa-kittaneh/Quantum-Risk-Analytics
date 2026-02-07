import time
import os
import logging
from flask import Flask, jsonify
from flask_cors import CORS

# Architecture Imports
from config import SCENARIOS, setup_logging
from services.data_service import DataService
from services.risk_service import RiskService, QuantumRiskEngine

# Initialize core services
setup_logging()
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Singleton service instances
quantum_engine = QuantumRiskEngine()
data_svc = DataService()
risk_svc = RiskService()

# ==================== CONTROLLER ROUTES ====================

@app.route('/', methods=['GET'])
def home():
    """Root endpoint for architectural status."""
    return jsonify({
        "status": "online",
        "architecture": "Controller-Service Pattern",
        "service": "Quantum Risk Analytics API",
        "version": "2.0 - Refactored"
    })

@app.route('/api/health', methods=['GET'])
def health():
    """System health check and dependency verification."""
    market_exists = os.path.exists('market_returns.csv')
    loans_exists = os.path.exists('loan_data.csv')
    
    return jsonify({
        "status": "healthy",
        "quantum_engine": "operational",
        "data_layer": {
            "market_returns": "present" if market_exists else "missing",
            "loan_data": "present" if loans_exists else "missing"
        }
    })

@app.route('/api/scenarios', methods=['GET'])
def get_scenarios():
    """Controller to return all available risk scenarios."""
    scenarios_list = [
        {
            "id": key,
            "name": cfg["name"],
            "description": cfg["description"],
            "volatility_factor": cfg["volatility_multiplier"],
            "risk_level": cfg["risk_level"]
        } for key, cfg in SCENARIOS.items()
    ]
    return jsonify({"success": True, "scenarios": scenarios_list})

@app.route('/api/quantum-simulation/<scenario>', methods=['GET'])
def run_simulation(scenario):
    """
    Main Controller for the risk simulation process.
    Orchestrates the data flow between DataService and RiskService.
    """
    if scenario not in SCENARIOS:
        return jsonify({"error": "Invalid scenario"}), 400
        
    config = SCENARIOS[scenario]
    start_t = time.time()
    
    # 1. Pipeline: Load Data (Controller asks Service)
    market_df, loan_df = data_svc.load_financial_data()
    
    # Safety: Fallback to mock data if core files are missing
    if market_df is None:
        market_df = data_svc.get_mock_market_data()

    # 2. Pipeline: Market Risk Calculation (Quantum Engine)
    sigma = data_svc.calculate_portfolio_volatility(market_df, config['volatility_multiplier'])
    q_result = quantum_engine.run_qae(sigma, config['mean_drift'])
    
    # 3. Pipeline: Credit Risk Analysis (Business Logic Service)
    c_result = risk_svc.calculate_credit_impact(loan_df, config.get('credit_stress_factor', 1.0))
    
    # 4. Pipeline: Final Integration
    risk_metrics = risk_svc.calculate_integrated_risk(q_result, c_result, sigma, config)
    
    # Helper variables for flattened response structure
    total_loss_val = risk_metrics['total_aggregated_risk']['total_economic_capital']
    risk_prob_val = risk_metrics['market_risk']['risk_probability']
    confidence_val = risk_metrics['market_risk']['quantum_confidence']
    
    # Constructing Unified Response
    response = {
        "status": "success",
        "scenario": config['name'],
        "execution_time": f"{time.time() - start_t:.4f}s",
        "estimated_loss": f"${total_loss_val:,.2f}",
        "risk_probability": f"{risk_prob_val}%",
        "confidence_level": f"{confidence_val}%",
        
        "scenario_details": {
            "name": config['name'],
            "risk_level": config['risk_level'],
            "intensity": f"{config['volatility_multiplier']}x"
        },
        "quantum_metrics": {
            "risk_probability": f"{risk_prob_val:.2f}%",
            "estimated_loss_dollars": f"${total_loss_val:,.0f}",
            "expected_loss_percentage": f"{risk_metrics['market_risk']['var_95_pct']:.2f}%",
            "confidence_level": f"{confidence_val:.1f}%",
            "execution_time": f"{time.time() - start_t:.4f}s",
            "quantum_estimation": f"{q_result.get('estimation', 0):.6f}",
            "oracle_queries": q_result.get('num_queries', 0)
        },
        "market_impact": {
            "gold_volatility": f"{market_df['Gold'].std() * config['volatility_multiplier'] * 100:.2f}%",
            "sp500_volatility": f"{market_df['SP500'].std() * config['volatility_multiplier'] * 100:.2f}%",
            "description": f"Quantum analysis for {config['name']} with Integrated Credit Risk."
        },
        "credit_impact": {
             "projected_defaults": risk_metrics['credit_risk']['estimated_defaults'],
             "stress_factor": f"{config.get('credit_stress_factor', 1.0)}x"
        },
        "risk_analysis": risk_metrics
    }
    
    return jsonify(response)

@app.route('/api/market-data', methods=['GET'])
def market_data():
    """Controller to return detailed market statistics."""
    try:
        market_df, _ = data_svc.load_financial_data()
        
        if market_df is None:
             market_df = data_svc.get_mock_market_data()

        # Calculation (Directly in controller for simple statistical views)
        gold_vol = market_df['Gold'].std()
        spy_vol = market_df['SP500'].std()
        gold_ret = market_df['Gold'].mean() * 252
        spy_ret = market_df['SP500'].mean() * 252
        correlation = market_df['Gold'].corr(market_df['SP500'])
        
        return jsonify({
            "success": True,
            "assets": {
                "gold": {
                    "symbol": "GLD",
                    "volatility": f"{gold_vol * 100:.2f}%",
                    "annual_return": f"{gold_ret * 100:.2f}%",
                    "status": "Stable" if gold_vol < 0.015 else "Volatile",
                    "direction": "↑" if gold_ret > 0 else "↓"
                },
                "spy": {
                    "symbol": "SPY",
                    "volatility": f"{spy_vol * 100:.2f}%",
                    "annual_return": f"{spy_ret * 100:.2f}%",
                    "status": "Stable" if spy_vol < 0.015 else "Volatile",
                    "direction": "↑" if spy_ret > 0 else "↓"
                }
            },
            "portfolio": {
                "correlation": round(correlation, 3),
                "total_days": len(market_df),
                "period": "Historical Crisis Data"
            }
        })
    except Exception as e:
        logger.error(f"Error in market_data controller: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting refactored Quantum Risk Controller...")
    app.run(debug=True, host='0.0.0.0', port=5000)
