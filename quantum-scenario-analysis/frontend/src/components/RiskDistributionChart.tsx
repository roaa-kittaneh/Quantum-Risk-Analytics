import {
  Area,
  AreaChart,
  ReferenceLine,
  ResponsiveContainer,
  XAxis,
  YAxis,
  Tooltip,
} from "recharts";

interface RiskDistributionChartProps {
  varThreshold: number;
  data?: Array<{
    x: number;
    normal: number;
    tail: number;
  }>;
}

export function RiskDistributionChart({ varThreshold, data = [] }: RiskDistributionChartProps) {
  // Use provided data or empty array if not yet available

  return (
    <div className="w-full h-64">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart
          data={data}
          margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
        >
          <defs>
            <linearGradient id="normalGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="hsl(217, 91%, 60%)" stopOpacity={0.3} />
              <stop offset="95%" stopColor="hsl(217, 91%, 60%)" stopOpacity={0.05} />
            </linearGradient>
            <linearGradient id="tailGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="hsl(0, 72%, 51%)" stopOpacity={0.5} />
              <stop offset="95%" stopColor="hsl(0, 72%, 51%)" stopOpacity={0.1} />
            </linearGradient>
          </defs>
          <XAxis
            dataKey="x"
            tickLine={false}
            axisLine={{ stroke: "hsl(214, 32%, 91%)" }}
            tick={{ fill: "hsl(215, 16%, 47%)", fontSize: 12 }}
            tickFormatter={(value) => `${Number(value).toFixed(1)}σ`}
          />
          <YAxis
            tickLine={false}
            axisLine={false}
            tick={{ fill: "hsl(215, 16%, 47%)", fontSize: 12 }}
            tickFormatter={() => ""}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "white",
              border: "1px solid hsl(214, 32%, 91%)",
              borderRadius: "8px",
              boxShadow: "0 4px 6px -1px rgb(0 0 0 / 0.1)",
            }}
            formatter={(value: number) => [value.toFixed(4), "Probability"]}
            labelFormatter={(label) => `${label}σ from mean`}
          />
          <ReferenceLine
            x={varThreshold}
            stroke="hsl(0, 72%, 51%)"
            strokeWidth={2}
            strokeDasharray="5 5"
            label={{
              value: "VaR Threshold",
              position: "top",
              fill: "hsl(0, 72%, 51%)",
              fontSize: 11,
              fontWeight: 500,
            }}
          />
          <Area
            type="monotone"
            dataKey="normal"
            stroke="hsl(217, 91%, 60%)"
            strokeWidth={2}
            fill="url(#normalGradient)"
          />
          {/* Overlay tail risk area */}
          <Area
            type="monotone"
            dataKey="tail"
            stroke="hsl(0, 72%, 51%)"
            strokeWidth={2}
            fill="url(#tailGradient)"
            connectNulls={false}
          />
        </AreaChart>
      </ResponsiveContainer>
      <div className="flex justify-center gap-6 mt-4 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-primary opacity-60" />
          <span className="text-muted-foreground">Normal Distribution</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-risk-red opacity-60" />
          <span className="text-muted-foreground">Tail Risk Zone</span>
        </div>
      </div>
    </div>
  );
}
