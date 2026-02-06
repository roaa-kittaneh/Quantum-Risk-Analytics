import { ArrowDown, ArrowUp } from "lucide-react";
import { cn } from "@/lib/utils";

interface AssetPerformanceCardProps {
    symbol: string;
    data: {
        total_return_pct: number;
        direction: "RISING" | "FALLING";
        volatility: number;
        risk_level: "LOW" | "MEDIUM" | "HIGH";
    };
}

export function AssetPerformanceCard({ symbol, data }: AssetPerformanceCardProps) {
    const isRising = data.direction === "RISING";
    const isHighRisk = data.risk_level === "HIGH";
    const isMediumRisk = data.risk_level === "MEDIUM";
    const isLowRisk = data.risk_level === "LOW";

    return (
        <div className="card-banking p-4 flex flex-col gap-3">
            <div className="flex items-center justify-between">
                <span className="font-bold text-lg">{symbol}</span>
                <span
                    className={cn(
                        "px-2 py-1 rounded text-xs font-medium border",
                        isHighRisk && "bg-red-100 text-red-700 border-red-200",
                        isMediumRisk && "bg-yellow-100 text-yellow-700 border-yellow-200",
                        isLowRisk && "bg-green-100 text-green-700 border-green-200"
                    )}
                >
                    {data.risk_level} RISK
                </span>
            </div>

            <div className="flex items-center gap-2">
                {isRising ? (
                    <div className="flex items-center text-green-600 gap-1">
                        <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                        <ArrowUp className="h-4 w-4" />
                    </div>
                ) : (
                    <div className="flex items-center text-red-600 gap-1">
                        <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                        <ArrowDown className="h-4 w-4" />
                    </div>
                )}
                <span className={cn("text-2xl font-bold", isRising ? "text-green-600" : "text-red-600")}>
                    {data.total_return_pct > 0 ? "+" : ""}
                    {data.total_return_pct.toFixed(1)}%
                </span>
            </div>

            <div className="text-xs text-muted-foreground mt-1">
                Volatility: {data.volatility.toFixed(1)}%
            </div>
        </div>
    );
}
