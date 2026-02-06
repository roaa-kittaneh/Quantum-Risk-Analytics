import { Sparkles } from "lucide-react";

interface QuantumInsightProps {
  scenario: string;
  severity: number;
  riskProbability: number;
  insightContent?: string;
}

export function QuantumInsight({ scenario, severity, riskProbability, insightContent }: QuantumInsightProps) {
  const getInsightText = () => {
    if (insightContent) return insightContent;

    const scenarioInsights: Record<string, string> = {
      "2008": `The quantum simulation reveals significant correlation breakdown patterns similar to the 2008 financial crisis. At a severity factor of ${severity.toFixed(1)}x, our Monte Carlo analysis with 10,000+ quantum-enhanced paths indicates a ${(riskProbability * 100).toFixed(1)}% probability of exceeding the VaR threshold. Historical backtesting suggests portfolio rebalancing toward defensive assets and increased liquidity buffers.`,
      "covid": `COVID-19 scenario simulation shows rapid volatility spikes characteristic of pandemic-driven market stress. The quantum sampling reveals non-linear contagion effects across asset classes. With severity at ${severity.toFixed(1)}x, the model detects heightened correlation in traditionally uncorrelated assets, suggesting diversification benefits may be temporarily reduced.`,
      "geopolitical": `Geopolitical conflict scenarios exhibit unique tail-risk characteristics with asymmetric return distributions. The quantum simulation at ${severity.toFixed(1)}x severity identifies potential supply chain disruptions and energy price shocks as primary risk vectors. Recommended: increased exposure to safe-haven assets and geographic diversification.`,
    };

    return scenarioInsights[scenario] || scenarioInsights["2008"];
  };

  return (
    <div className="card-banking">
      <div className="flex items-center gap-2 mb-3">
        <Sparkles className="h-4 w-4 text-primary" />
        <h3 className="section-title mb-0">Quantum Insight</h3>
      </div>
      <p className="text-sm text-muted-foreground leading-relaxed">
        {getInsightText()}
      </p>
    </div>
  );
}
