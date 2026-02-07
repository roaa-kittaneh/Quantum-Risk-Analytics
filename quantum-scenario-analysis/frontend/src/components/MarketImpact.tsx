import { Shield, Coins, TrendingDown, AlertTriangle } from "lucide-react";
import { ImpactData } from "@/types";

interface MarketImpactProps {
    impactData: ImpactData;
}

/**
 * Displays key market impact metrics (Gold, S&P 500, Risk Ratio).
 * Includes creative styling and layout for insights.
 *
 * @component
 * @example
 * <MarketImpact impactData={data} />
 */
export const MarketImpact = ({ impactData }: MarketImpactProps) => {
    return (
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
                                <span className="text-2xl font-bold text-foreground">
                                    +/- {impactData.goldImpact}%
                                </span>
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
                                <span className="text-2xl font-bold text-foreground">
                                    +/- {impactData.spyImpact}%
                                </span>
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
    );
};
