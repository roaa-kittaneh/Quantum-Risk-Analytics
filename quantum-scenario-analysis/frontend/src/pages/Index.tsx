import { useState, useEffect } from "react";
import { Activity, TrendingDown, Shield, BarChart3, Zap, Coins, AlertTriangle, Gauge } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ExplanationDiagram } from "@/components/ExplanationDiagram";
import { KPICard } from "@/components/KPICard";
import { AssetBadge } from "@/components/AssetBadge";
import { useToast } from "@/hooks/use-toast";
import { cn } from "@/lib/utils";

// Updated Interface to match the simplified Backend Response
interface SimulationResult {
  status: string;
  scenario: string;
  estimated_loss_dollars: string;
  estimated_loss_percentage: string;
  confidence_level: string;
  risk_probability: string;
  quantum_execution_time: string;
  market_impact: string;
}

interface MarketData {
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

const Index = () => {
  const { toast } = useToast();
  const [selectedScenario, setSelectedScenario] = useState("baseline");
  const [marketData, setMarketData] = useState<MarketData | null>(null);
  const [isSimulating, setIsSimulating] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [results, setResults] = useState<SimulationResult | null>(null);

  // Parse volatility strings "15.5%" -> 0.155
  const parseVol = (volStr: string | undefined) => {
    if (!volStr) return 0;
    return parseFloat(volStr.replace("%", "")) / 100;
  };

  // Fetch market data on load
  useEffect(() => {
    fetch("http://localhost:5000/api/market-data")
      .then((res) => res.json())
      .then((data) => setMarketData(data))
      .catch((err) => console.error("Failed to fetch market data", err));
  }, []);

  const runSimulation = async () => {
    setIsSimulating(true);
    setShowResults(false);

    try {
      const response = await fetch(`http://localhost:5000/api/quantum-simulation/${selectedScenario}`);

      if (!response.ok) {
        throw new Error("Failed to connect to the backend");
      }

      const data: SimulationResult = await response.json();

      if (data.status === "success") {
        setResults(data);
        setShowResults(true);
        toast({
          title: "Quantum Analysis Complete",
          description: `Scenario: ${data.scenario}`,
        });
      } else {
        throw new Error("Analysis failed");
      }
    } catch (error) {
      console.error("Simulation error:", error);
      toast({
        title: "Connection Failed",
        description: "Could not connect to the quantum backend.",
        variant: "destructive",
      });
    } finally {
      setIsSimulating(false);
    }
  };

  // Helper to calculate impacts for the Insights Grid
  const getImpacts = () => {
    if (!results || !marketData) return null;

    // Determine intensity locally since backend structure changed
    // Determine intensity using selectedScenario
    const intensityMap: Record<string, number> = {
      "mild": 0.5,
      "baseline": 1.0,
      "super": 1.5
    };
    const intensity = intensityMap[selectedScenario] || 1.0;

    const goldSigma = parseVol(marketData.assets.gold.volatility);
    const spySigma = parseVol(marketData.assets.spy.volatility);

    return {
      goldImpact: (goldSigma * intensity * 1.96 * 100).toFixed(2),
      spyImpact: (spySigma * intensity * 1.96 * 100).toFixed(2),
      riskRatio: (spySigma / goldSigma).toFixed(1),
      intensity,
      goldSigma,
      spySigma
    };
  };

  const impactData = getImpacts();

  // Helper to determine Risk Level based on scenario
  const getRiskLevel = (scenario: string) => {
    if (scenario === "Future Super-Crisis") return "EXTREME";
    if (scenario === "Baseline Crisis") return "HIGH";
    return "MODERATE";
  };

  // Dynamic Styles
  const getRiskColor = (level: string) => {
    switch (level) {
      case "EXTREME": return "text-red-500 shadow-red-500/50";
      case "HIGH": return "text-orange-500 shadow-orange-500/50";
      default: return "text-green-500 shadow-green-500/50";
    }
  };

  const getRiskBg = (level: string) => {
    switch (level) {
      case "EXTREME": return "bg-red-500/10 border-red-500/20";
      case "HIGH": return "bg-orange-500/10 border-orange-500/20";
      default: return "bg-green-500/10 border-green-500/20";
    }
  };

  const currentRiskLevel = results ? getRiskLevel(results.scenario) : "MODERATE";

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
              <h1 className="text-lg font-semibold text-foreground">Quantum Risk Analytics</h1>
              <p className="text-xs text-muted-foreground">Advanced Portfolio Stress Testing</p>
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

          {/* Scenario Selector */}
          <div className="mt-6 grid gap-4">
            <label className="text-sm font-medium text-foreground">Select Crisis Scenario</label>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {[
                { id: "mild", label: "Mild Disruption", desc: "Low Volatility (0.5x)" },
                { id: "baseline", label: "Baseline Crisis", desc: "COVID-19 Scale (1.0x)" },
                { id: "super", label: "Future Super-Crisis", desc: "Extreme Event (1.5x)" },
              ].map((scenario) => (
                <button
                  key={scenario.id}
                  onClick={() => setSelectedScenario(scenario.id)}
                  className={cn(
                    "relative flex flex-col items-start p-4 rounded-lg border transition-all duration-200 text-left",
                    selectedScenario === scenario.id
                      ? "border-primary bg-primary/5 shadow-md"
                      : "border-border hover:border-primary/50 hover:bg-muted/50"
                  )}
                >
                  <span className="font-semibold text-sm">{scenario.label}</span>
                  <span className="text-xs text-muted-foreground mt-1">{scenario.desc}</span>
                  {selectedScenario === scenario.id && (
                    <div className="absolute top-3 right-3 h-2 w-2 rounded-full bg-primary animate-pulse" />
                  )}
                </button>
              ))}
            </div>
          </div>

          {/* Run Button with Loading Overlay */}
          <div className="mt-8 relative">
            <Button
              variant="quantum"
              size="lg"
              onClick={runSimulation}
              disabled={isSimulating}
              className="w-full md:w-auto min-w-[200px]"
            >
              {isSimulating ? (
                <>
                  <Activity className="h-4 w-4 animate-spin mr-2" />
                  Running QAE Algorithm...
                </>
              ) : (
                <>
                  <Zap className="h-4 w-4 mr-2" />
                  Run Deployment
                </>
              )}
            </Button>

            {isSimulating && (
              <div className="absolute top-full left-0 mt-3 text-xs text-muted-foreground animate-pulse">
                Quantum circuits initializing... Calculating amplitude estimation...
              </div>
            )}
          </div>
        </section>

        {/* Results Section */}
        {showResults && results && impactData && (
          <section className="space-y-6 animate-slide-up">

            {/* Scenario Status Bar */}
            <div className={`rounded-lg border p-4 flex items-center justify-between ${getRiskBg(currentRiskLevel)}`}>
              <div className="flex items-center gap-3">
                <Gauge className={`h-6 w-6 ${getRiskColor(currentRiskLevel).split(" ")[0]}`} />
                <div>
                  <h4 className="font-bold text-foreground capitalize">{results.scenario} Scenario Detected</h4>
                  <p className="text-xs text-muted-foreground">{results.market_impact}</p>
                </div>
              </div>
              <div className={`px-3 py-1 rounded-full text-xs font-bold border ${getRiskBg(currentRiskLevel)} ${getRiskColor(currentRiskLevel).split(" ")[0]}`}>
                RISK LEVEL: {currentRiskLevel}
              </div>
            </div>

            {/* KPI Cards */}
            <div className="grid gap-4 md:grid-cols-3">
              <KPICard
                label="Risk Probability"
                value={results.risk_probability}
                subtext="Probability of Crisis Event"
                variant={currentRiskLevel === "EXTREME" ? "destructive" : currentRiskLevel === "HIGH" ? "warning" : "success"}
              />
              <KPICard
                label="Estimated Loss"
                value={results.estimated_loss_dollars}
                subtext={`Potential Portfolio Impact (${results.estimated_loss_percentage})`}
                variant={currentRiskLevel === "EXTREME" ? "destructive" : currentRiskLevel === "HIGH" ? "warning" : "success"}
              />
              <KPICard
                label="Confidence Level"
                value={results.confidence_level}
                subtext={`Verified in ${results.quantum_execution_time}`}
                variant="success"
              />
            </div>

            {/* CREATIVE INSIGHTS GRID */}
            <div className="relative overflow-hidden rounded-xl border border-border bg-gradient-to-br from-card to-background shadow-lg">
              <div className="absolute -top-24 -right-24 h-48 w-48 rounded-full bg-primary/10 blur-3xl" />

              <div className="p-6">
                <h3 className="section-title mb-6 flex items-center gap-2 relative z-10">
                  <Shield className="h-5 w-5 text-primary" />
                  <span className="bg-clip-text text-transparent bg-gradient-to-r from-primary to-blue-600 font-bold">
                    MARKET IMPACT ANALYSIS
                  </span>
                </h3>

                <div className="grid gap-4 md:grid-cols-3 relative z-10">
                  {/* Gold */}
                  <div className="group relative overflow-hidden rounded-lg bg-background/40 p-4 border border-amber-200/20 hover:border-amber-400/40 transition-all duration-300">
                    <div className="absolute inset-0 bg-amber-500/5 group-hover:bg-amber-500/10 transition-colors" />
                    <div className="flex items-start gap-4">
                      <div className="rounded-full bg-amber-500/10 p-2.5">
                        <Coins className="h-5 w-5 text-amber-500" />
                      </div>
                      <div>
                        <p className="text-sm font-medium text-muted-foreground">Gold Fluctuation</p>
                        <span className="text-2xl font-bold text-foreground">+/- {impactData.goldImpact}%</span>
                      </div>
                    </div>
                  </div>

                  {/* S&P 500 */}
                  <div className="group relative overflow-hidden rounded-lg bg-background/40 p-4 border border-blue-200/20 hover:border-blue-400/40 transition-all duration-300">
                    <div className="absolute inset-0 bg-blue-500/5 group-hover:bg-blue-500/10 transition-colors" />
                    <div className="flex items-start gap-4">
                      <div className="rounded-full bg-blue-500/10 p-2.5">
                        <TrendingDown className="h-5 w-5 text-blue-500" />
                      </div>
                      <div>
                        <p className="text-sm font-medium text-muted-foreground">S&P 500 Fluctuation</p>
                        <span className="text-2xl font-bold text-foreground">+/- {impactData.spyImpact}%</span>
                      </div>
                    </div>
                  </div>

                  {/* Risk Ratio */}
                  <div className="group relative overflow-hidden rounded-lg bg-background/40 p-4 border border-red-200/20 hover:border-red-400/40 transition-all duration-300">
                    <div className="absolute inset-0 bg-red-500/5 group-hover:bg-red-500/10 transition-colors" />
                    <div className="flex items-start gap-4">
                      <div className="rounded-full bg-red-500/10 p-2.5">
                        <AlertTriangle className="h-5 w-5 text-red-500" />
                      </div>
                      <div>
                        <p className="text-sm font-medium text-muted-foreground">Risk Assessment</p>
                        <span className="text-2xl font-bold text-foreground">{impactData.riskRatio}x</span>
                        <span className="text-xs text-muted-foreground ml-1">Riskier than Gold</span>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="mt-6 pt-4 border-t border-border/50 text-xs text-muted-foreground flex items-center gap-2">
                  <div className="h-1.5 w-1.5 rounded-full bg-green-500 animate-pulse" />
                  Dynamic Volatility Adjustment: {impactData.intensity}x Intensity
                </div>
              </div>
            </div>

            {/* Explanation Diagram */}
            <ExplanationDiagram
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
