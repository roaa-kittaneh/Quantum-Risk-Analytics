import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Area,
    ComposedChart,
    ReferenceLine,
    Label
} from "recharts";

interface QuantumChartProps {
    goldSigma: number;
    sp500Sigma: number;
    riskRatio: number;
    intensity: number;
}

/**
 * Renders a comparison chart between Normal Distribution and Crisis Model (Fat Tails).
 * Visualizes the difference in probability density for extreme events.
 *
 * @component
 * @example
 * <QuantumChart goldSigma={0.15} sp500Sigma={0.25} riskRatio={1.6} intensity={1.5} />
 */
export const QuantumChart = ({ goldSigma, sp500Sigma, riskRatio, intensity = 1.0 }: QuantumChartProps) => {

    // Generate data for the chart
    const generateChartData = () => {
        const data = [];
        const mean = 0;
        const stdDev = 0.02; // Standard deviation for basic normal curve

        // Dynamic widening factor for tails based on intensity
        // mild (0.5) -> narrower tails
        // baseline (1.0) -> normal fat tails
        // stronger (1.5) -> super wide tails
        const tailWidth = 2.5 * Math.max(0.8, intensity);
        const tailWeight = 0.3 * Math.min(1.5, intensity); // Heavier tails in crisis

        for (let x = -0.15; x <= 0.15; x += 0.005) {
            // Normal Distribution (Blue Line)
            const normalY = (1 / (stdDev * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((x - mean) / stdDev, 2));

            // Crisis Model / Fat Tail (Red Line)
            const fatTailY = (1 / ((stdDev * 0.6) * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((x - mean) / (stdDev * 0.6), 2)) * (1 - tailWeight)
                + (1 / (stdDev * tailWidth * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((x - mean) / (stdDev * tailWidth), 2)) * tailWeight;

            // Identify "Crisis Zone" for shading (expand zone with intensity)
            const threshold = 0.05 / Math.sqrt(intensity); // Zone starts earlier in high intensity
            const crisisZone = (x < -threshold || x > threshold) ? fatTailY : 0;

            data.push({
                xOffset: x,
                xLabel: x.toFixed(2),
                normal: normalY,
                crisis: fatTailY,
                crisisZone: crisisZone
            });
        }
        return data;
    };

    const data = generateChartData();

    return (
        <div className="card-banking p-6 space-y-6">
            <div className="text-center">
                <h3 className="text-lg font-bold text-foreground">Why Normal Distribution Fails in "Hard Situations"</h3>
                <p className="text-xs text-muted-foreground mt-1">Comparing Traditional Models vs. Your Quantum Fat-Tail Model</p>
            </div>

            <div className="h-[350px] w-full relative">
                <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={data} margin={{ top: 20, right: 20, bottom: 20, left: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" opacity={0.2} vertical={true} />
                        <XAxis
                            dataKey="xLabel"
                            label={{ value: 'Market Returns / Losses', position: 'bottom', offset: 0 }}
                            tick={{ fontSize: 10 }}
                            interval={4}
                        />
                        <YAxis
                            label={{ value: 'Probability Density', angle: -90, position: 'insideLeft' }}
                            tick={false}
                        />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#f8fafc' }}
                            labelStyle={{ color: '#94a3b8' }}
                        />

                        {/* Shaded Area for Extreme Risk (Red zones) */}
                        <Area
                            type="monotone"
                            dataKey="crisisZone"
                            fill="#ef4444"
                            fillOpacity={0.3}
                            stroke="none"
                        />

                        {/* Normal Distribution - Blue Dashed */}
                        <Line
                            type="monotone"
                            dataKey="normal"
                            stroke="#3b82f6"
                            strokeWidth={2}
                            strokeDasharray="5 5"
                            dot={false}
                            name="Normal Distribution (Fails in Crises)"
                        />

                        {/* Crisis Model - Red Solid */}
                        <Line
                            type="monotone"
                            dataKey="crisis"
                            stroke="#dc2626"
                            strokeWidth={3}
                            dot={false}
                            name="Crisis Model (Fat Tails - Our Project)"
                        />

                        {/* Annotation for Crisis Zone */}
                        <ReferenceLine x="20" stroke="none" label="Crisis Zone" />
                    </ComposedChart>
                </ResponsiveContainer>

                {/* Custom Annotations mimicking the image */}
                <div className="absolute top-[60%] left-[15%] text-xs font-bold text-red-600">
                    Crisis Zone:<br />Extreme Losses
                    <div className="text-xl">↓</div>
                </div>

                <div className="absolute top-4 right-4 bg-white/90 dark:bg-slate-800/90 p-2 rounded border border-border text-xs shadow-sm">
                    <div className="flex items-center gap-2 mb-1">
                        <div className="w-4 h-0.5 bg-blue-500 border-dashed border-t-2 border-blue-500"></div>
                        <span>Normal Distribution (Fails in Crises)</span>
                    </div>
                    <div className="flex items-center gap-2 mb-1">
                        <div className="w-4 h-1 bg-red-600 rounded"></div>
                        <span>Crisis Model (Fat Tails - Our Project)</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-4 h-3 bg-red-500/30 rounded"></div>
                        <span>Extreme Risk (Black Swans)</span>
                    </div>
                </div>
            </div>

            <div className="text-xs text-muted-foreground text-center border-t pt-4">
                This chart shows how your project's model (Red) captures the probability of extreme events (Fat Tails) that standard models (Blue) practically ignore (probability ~0).
            </div>
        </div>
    );
};
