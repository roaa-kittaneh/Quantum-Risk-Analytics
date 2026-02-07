import { Activity, Shield } from "lucide-react";
import { KPICard } from "@/components/KPICard";
import { RiskCard } from "@/components/RiskCard";
import { ScenarioSelector } from "@/components/ScenarioSelector";
import { MarketImpact } from "@/components/MarketImpact";
import { QuantumChart } from "@/components/QuantumChart";
import { useQuantumApi } from "@/hooks/useQuantumApi";
import { RISK_LEVELS } from "@/constants";

const Index = () => {
  const {
    selectedScenario,
    setSelectedScenario,
    runSimulation,
    isSimulating,
    results,
    showResults,
    impactData,
    riskLevel,
  } = useQuantumApi();

  const getKPIVariant = (level: string) => {
    switch (level) {
      case RISK_LEVELS.EXTREME:
        return "destructive";
      case RISK_LEVELS.HIGH:
        return "warning";
      default:
        return "success";
    }
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card">
        <div className="container py-4">
          <div className="flex items-center gap-3">
            <div className="h-8 w-8 rounded-lg bg-primary flex items-center justify-center">
              <Activity className="h-4 w-4 text-primary-foreground" />
            </div>
            <div>
              <h1 className="text-lg font-semibold text-foreground">
                Quantum Risk Analytics
              </h1>
              <p className="text-xs text-muted-foreground">
                Advanced Portfolio Stress Testing
              </p>
            </div>
          </div>
        </div>
      </header>

      <main className="container py-8 space-y-8">
        {/* Configuration Section */}
        <section className="card-banking">
          <h2 className="section-title flex items-center gap-2">
            <Shield className="h-5 w-5 text-primary" />
            Scenario Configuration
          </h2>

          <ScenarioSelector
            selectedScenario={selectedScenario}
            onScenarioChange={setSelectedScenario}
            onRunSimulation={runSimulation}
            isSimulating={isSimulating}
          />
        </section>

        {/* Results Section */}
        {showResults && results && impactData && (
          <section className="space-y-6 animate-slide-up">
            {/* Scenario Status Bar */}
            <RiskCard
              scenario={results.scenario_details.name}
              marketImpact={results.market_impact.description}
              riskLevel={riskLevel}
            />

            {/* KPI Cards */}
            <div className="grid gap-4 md:grid-cols-3">
              <KPICard
                label="Risk Probability"
                value={results.quantum_metrics.risk_probability}
                subtext="Probability of Crisis Event"
                variant={getKPIVariant(riskLevel)}
              />
              <KPICard
                label="Estimated Loss"
                value={results.quantum_metrics.estimated_loss_dollars}
                subtext={`Potential Portfolio Impact (${results.quantum_metrics.expected_loss_percentage})`}
                variant={getKPIVariant(riskLevel)}
              />
              <KPICard
                label="Confidence Level"
                value={results.quantum_metrics.confidence_level}
                subtext={`Verified in ${results.quantum_metrics.execution_time}`}
                variant="success"
              />
            </div>

            {/* CREATIVE INSIGHTS GRID */}
            <MarketImpact impactData={impactData} />

            {/* Explanation Diagram */}
            <QuantumChart
              goldSigma={impactData.goldSigma}
              sp500Sigma={impactData.spySigma}
              riskRatio={parseFloat(impactData.riskRatio)}
              intensity={impactData.intensity}
            />
          </section>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-border bg-card mt-12">
        <div className="container py-4">
          <p className="text-xs text-muted-foreground text-center">
            Quantum Risk Analytics Platform • Powered by Qiskit Aer & Flask
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
