import { describe, it, expect } from "vitest";
import { DEVIATION_CONFIG } from "../constants";

describe("DEVIATION_CONFIG", () => {
  it("should have correct SMA period", () => {
    expect(DEVIATION_CONFIG.SMA_PERIOD).toBe(200);
  });

  it("should have ordered risk levels", () => {
    const { RISK_LEVELS } = DEVIATION_CONFIG;
    expect(RISK_LEVELS.LOW).toBeLessThan(RISK_LEVELS.MODERATE);
    expect(RISK_LEVELS.MODERATE).toBeLessThan(RISK_LEVELS.HIGH);
  });

  it("should have positive refresh interval", () => {
    expect(DEVIATION_CONFIG.REFRESH_INTERVAL_MS).toBeGreaterThan(0);
  });

  it("simulation parameters should be between 0 and 1", () => {
    const { SIMULATION } = DEVIATION_CONFIG;
    expect(SIMULATION.VOLATILITY).toBeGreaterThan(0);
    expect(SIMULATION.VOLATILITY).toBeLessThan(1);
    expect(SIMULATION.MEAN_REVERSION).toBeGreaterThan(0);
    expect(SIMULATION.MEAN_REVERSION).toBeLessThan(1);
  });
});
