import { Gauge } from "lucide-react";
import { RISK_LEVELS } from "@/constants";

interface RiskCardProps {
    scenario: string;
    marketImpact: string;
    riskLevel: string;
}

const getRiskColor = (level: string) => {
    switch (level) {
        case RISK_LEVELS.EXTREME:
            return "text-red-500 shadow-red-500/50";
        case RISK_LEVELS.HIGH:
            return "text-orange-500 shadow-orange-500/50";
        default:
            return "text-green-500 shadow-green-500/50";
    }
};

const getRiskBg = (level: string) => {
    switch (level) {
        case RISK_LEVELS.EXTREME:
            return "bg-red-500/10 border-red-500/20";
        case RISK_LEVELS.HIGH:
            return "bg-orange-500/10 border-orange-500/20";
        default:
            return "bg-green-500/10 border-green-500/20";
    }
};

/**
 * Displays the current risk level status and scenario summary.
 * Color-coded based on the risk level.
 *
 * @component
 * @example
 * <RiskCard
 *   scenario="Baseline Disruption"
 *   marketImpact="-20%"
 *   riskLevel="HIGH"
 * />
 */
export const RiskCard = ({ scenario, marketImpact, riskLevel }: RiskCardProps) => {
    return (
        <div
            className={`rounded-lg border p-4 flex items-center justify-between ${getRiskBg(
                riskLevel
            )}`}
        >
            <div className="flex items-center gap-3">
                <Gauge
                    className={`h-6 w-6 ${getRiskColor(riskLevel).split(" ")[0]}`}
                />
                <div>
                    <h4 className="font-bold text-foreground capitalize">
                        {scenario} Scenario Detected
                    </h4>
                    <p className="text-xs text-muted-foreground">{marketImpact}</p>
                </div>
            </div>
            <div
                className={`px-3 py-1 rounded-full text-xs font-bold border ${getRiskBg(
                    riskLevel
                )} ${getRiskColor(riskLevel).split(" ")[0]}`}
            >
                RISK LEVEL: {riskLevel}
            </div>
        </div>
    );
};
