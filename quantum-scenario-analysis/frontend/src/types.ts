export interface MarketData {
    success: boolean;
    assets: {
        gold: {
            symbol: string;
            volatility: string;
            annual_return: string;
            status: string;
            direction: string;
        };
        spy: {
            symbol: string;
            volatility: string;
            annual_return: string;
            status: string;
            direction: string;
        };
    };
    portfolio: {
        correlation: number;
        total_days: number;
        period: string;
    };
}

export interface SimulationResult {
    status: string;
    scenario_details: {
        name: string;
        intensity: string;
        risk_level: string;
    };
    quantum_metrics: {
        expected_loss_percentage: string;
        estimated_loss_dollars: string;
        risk_probability: string;
        confidence_level: string;
        execution_time: string;
    };
    market_impact: {
        gold_fluctuation: string;
        sp500_fluctuation: string;
        description: string;
    };
    // Adding previous flattened fields for backward compatibility if any components still use them
    // Or better, update hook to flatten them. Let's keep the type reflecting the RAW response first.
}

export interface ImpactData {
    goldImpact: string;
    spyImpact: string;
    riskRatio: string;
    intensity: number;
    goldSigma: number;
    spySigma: number;
}
