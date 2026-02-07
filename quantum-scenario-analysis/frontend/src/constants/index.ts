export const API_BASE_URL = "http://localhost:5000/api";

export const SCENARIO_LABELS = {
    MILD: "Mild Disruption",
    BASELINE: "Baseline Crisis",
    SUPER: "Future Super-Crisis",
};

export const SCENARIOS = [
    { id: "mild", label: SCENARIO_LABELS.MILD, desc: "Low Volatility (0.5x)" },
    { id: "baseline", label: SCENARIO_LABELS.BASELINE, desc: "COVID-19 Scale (1.0x)" },
    { id: "super", label: SCENARIO_LABELS.SUPER, desc: "Extreme Event (1.5x)" },
];

export const INTENSITY_MAP: Record<string, number> = {
    mild: 0.5,
    baseline: 1.0,
    super: 1.5,
};

export const RISK_LEVELS = {
    EXTREME: "EXTREME",
    HIGH: "HIGH",
    MODERATE: "MODERATE",
};
