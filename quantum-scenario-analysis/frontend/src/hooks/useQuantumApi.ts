import { useState, useEffect, useMemo } from "react";
import { API_BASE_URL, INTENSITY_MAP, SCENARIO_LABELS } from "@/constants";
import { useToast } from "@/hooks/use-toast";
import { MarketData, SimulationResult, ImpactData } from "@/types";

/**
 * Custom hook to manage Quantum Simulation state and data fetching.
 * Handles scenario selection, running simulations, and calculating impact metrics.
 *
 * @returns {Object} Object containing state and handlers
 */
export const useQuantumApi = () => {
    const { toast } = useToast();
    const [selectedScenario, setSelectedScenario] = useState("baseline");
    const [marketData, setMarketData] = useState<MarketData | null>(null);
    const [isSimulating, setIsSimulating] = useState(false);
    const [results, setResults] = useState<SimulationResult | null>(null);
    const [showResults, setShowResults] = useState(false);

    useEffect(() => {
        fetch(`${API_BASE_URL}/market-data`)
            .then((res) => res.json())
            .then((data) => {
                if (data.success && data.assets) {
                    setMarketData(data);
                } else {
                    // Handle case where backend might return old structure or error
                    if (data.assets) {
                        setMarketData(data);
                    } else if (data.gold && data.spy) {
                        // Fallback for very old structure if backend wasn't updated
                        // But we just updated backend.
                        setMarketData({
                            success: true,
                            assets: {
                                gold: { ...data.gold, symbol: 'GLD', annual_return: '0%', status: 'Unknown', direction: 'Flat' },
                                spy: { ...data.spy, symbol: 'SPY', annual_return: '0%', status: 'Unknown', direction: 'Flat' }
                            },
                            portfolio: { correlation: 0, total_days: 0, period: 'Unknown' }
                        });
                    } else {
                        console.error("Invalid market data format", data);
                    }
                }
            })
            .catch((err) => console.error("Failed to fetch market data", err));
    }, []);

    const runSimulation = async () => {
        setIsSimulating(true);
        setShowResults(false);

        try {
            const response = await fetch(`${API_BASE_URL}/quantum-simulation/${selectedScenario}`);

            if (!response.ok) {
                throw new Error("Failed to connect to the backend");
            }

            const data: SimulationResult = await response.json();

            if (data.status === "success") {
                setResults(data);
                setShowResults(true);
                toast({
                    title: "Quantum Analysis Complete",
                    description: `Scenario: ${data.scenario_details.name}`,
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

    const parseVol = (volStr: string | undefined) => {
        if (!volStr) return 0;
        return parseFloat(volStr.replace("%", "")) / 100;
    };

    const impactData: ImpactData | null = useMemo(() => {
        if (!results || !marketData) return null;

        const intensity = INTENSITY_MAP[selectedScenario] || 1.0;
        // Ensure marketData.assets exists before accessing properties
        if (!marketData.assets?.gold || !marketData.assets?.spy) return null;

        const goldSigma = parseVol(marketData.assets.gold.volatility);
        const spySigma = parseVol(marketData.assets.spy.volatility);

        return {
            goldImpact: (goldSigma * intensity * 1.96 * 100).toFixed(2),
            spyImpact: (spySigma * intensity * 1.96 * 100).toFixed(2),
            riskRatio: (goldSigma !== 0 ? (spySigma / goldSigma).toFixed(1) : "0.0"),
            intensity,
            goldSigma,
            spySigma,
        };
    }, [results, marketData, selectedScenario]);

    const riskLevel = useMemo(() => {
        if (!results) return "MODERATE";
        const scenarioName = results.scenario_details?.name;

        // Map backend returned name to SCENARIO_LABELS if needed, or just check content
        // Backend returns "Baseline", "Future super-crisis" (capitalized).
        // SCENARIO_LABELS are "Mild Disruption", "Baseline Crisis", "Future Super-Crisis".

        // Simpler logic based on intensity or returned risk level directly from backend
        if (results.scenario_details?.risk_level) {
            return results.scenario_details.risk_level;
        }

        return "MODERATE";
    }, [results]);

    return {
        marketData,
        runSimulation,
        isSimulating,
        results,
        showResults,
        impactData,
        selectedScenario,
        setSelectedScenario,
        riskLevel,
    };
};
