import { cn } from "@/lib/utils";

interface KPICardProps {
  label: string;
  value: string;
  subtext?: string;
  variant?: "default" | "warning" | "success" | "destructive";
  className?: string;
}

export function KPICard({ label, value, subtext, variant = "default", className }: KPICardProps) {
  const valueColorClass = {
    default: "text-foreground",
    warning: "text-warning",
    destructive: "text-destructive",
    success: "text-success",
  }[variant];

  return (
    <div className={cn("card-banking flex flex-col", className)}>
      <span className="kpi-label mb-2">{label}</span>
      <span className={cn("kpi-value", valueColorClass)}>{value}</span>
      {subtext && (
        <span className="text-xs text-muted-foreground mt-1">{subtext}</span>
      )}
    </div>
  );
}
