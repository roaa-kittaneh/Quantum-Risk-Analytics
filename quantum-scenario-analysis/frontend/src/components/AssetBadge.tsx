import { cn } from "@/lib/utils";
import { ArrowUpRight, ArrowDownRight } from "lucide-react";

interface AssetBadgeProps {
  symbol: string;
  trend?: "up" | "down";
  className?: string;
}

export function AssetBadge({ symbol, trend, className }: AssetBadgeProps) {
  return (
    <span className={cn(
      "badge-asset flex items-center gap-1",
      trend === "up" && "bg-green-500/10 text-green-500 border-green-500/20",
      trend === "down" && "bg-red-500/10 text-red-500 border-red-500/20",
      className
    )}>
      {symbol}
      {trend === "up" && <ArrowUpRight className="h-3 w-3" />}
      {trend === "down" && <ArrowDownRight className="h-3 w-3" />}
    </span>
  );
}
