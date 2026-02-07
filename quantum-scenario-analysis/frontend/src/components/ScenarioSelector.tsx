import { Zap, Activity } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { SCENARIOS } from "@/constants";

interface ScenarioSelectorProps {
    selectedScenario: string;
    onScenarioChange: (scenarioId: string) => void;
    onRunSimulation: () => void;
    isSimulating: boolean;
}

/**
 * Component for selecting the crisis scenario to simulate.
 * Displays a grid of scenario options with descriptions.
 *
 * @component
 * @example
 * <ScenarioSelector
 *   selectedScenario="baseline"
 *   onScenarioChange={handleChange}
 *   onRunSimulation={handleRun}
 *   isSimulating={false}
 * />
 */
export const ScenarioSelector = ({
    selectedScenario,
    onScenarioChange,
    onRunSimulation,
    isSimulating,
}: ScenarioSelectorProps) => {
    return (
        <div className="mt-6 grid gap-4">
            <label className="text-sm font-medium text-foreground">Select Crisis Scenario</label>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                {SCENARIOS.map((scenario) => (
                    <button
                        key={scenario.id}
                        onClick={() => onScenarioChange(scenario.id)}
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

            <div className="mt-8 relative">
                <Button
                    variant="quantum"
                    size="lg"
                    onClick={onRunSimulation}
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
        </div>
    );
};
