export const API_BASE_URL = "http://localhost:5000/api";

export const SCENARIO_LABELS = {
    MILD: "Mild Disruption",
    BASELINE: "Baseline Crisis",
    SUPER: "Future Super-Crisis",
};

export const SCENARIOS = [
    { id: "mild", label: SCENARIO_LABELS.MILD, desc: "Correction (0.8x)" },
    { id: "baseline", label: SCENARIO_LABELS.BASELINE, desc: "Financial Crisis (1.5x)" },
    { id: "super", label: SCENARIO_LABELS.SUPER, desc: "Extreme Event (4.0x)" },
];

export const INTENSITY_MAP: Record<string, number> = {
    mild: 0.8,
    baseline: 1.5,
    super: 4.0,
};

export const RISK_LEVELS = {
    EXTREME: "EXTREME",
    HIGH: "HIGH",
    MODERATE: "MODERATE",
};
