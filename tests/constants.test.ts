import { describe, it, expect } from "vitest";
import { DEVIATION_CONFIG, INDICATOR_META, BUBBLE_REGIME_CONFIG } from "../constants";

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

describe("INDICATOR_META", () => {
  it("should have 6 entries", () => {
    expect(INDICATOR_META).toHaveLength(6);
  });

  it("each entry should have key, label, color, description, and category", () => {
    for (const indicator of INDICATOR_META) {
      expect(indicator).toHaveProperty("key");
      expect(indicator).toHaveProperty("label");
      expect(indicator).toHaveProperty("color");
      expect(indicator).toHaveProperty("description");
      expect(indicator).toHaveProperty("category");
    }
  });

  it("categories should be 'sentiment' or 'liquidity'", () => {
    for (const indicator of INDICATOR_META) {
      expect(["sentiment", "liquidity"]).toContain(indicator.category);
    }
  });

  it("all keys should be unique", () => {
    const keys = INDICATOR_META.map((i) => i.key);
    expect(new Set(keys).size).toBe(keys.length);
  });
});

describe("BUBBLE_REGIME_CONFIG", () => {
  it("should have 5 entries", () => {
    expect(BUBBLE_REGIME_CONFIG).toHaveLength(5);
  });

  it("thresholds should be ascending (30, 50, 70, 85, 100)", () => {
    const thresholds = BUBBLE_REGIME_CONFIG.map((r) => r.threshold);
    expect(thresholds).toEqual([30, 50, 70, 85, 100]);
  });

  it("each entry should have key, label, color, and description", () => {
    for (const regime of BUBBLE_REGIME_CONFIG) {
      expect(regime).toHaveProperty("key");
      expect(regime).toHaveProperty("label");
      expect(regime).toHaveProperty("color");
      expect(regime).toHaveProperty("description");
    }
  });
});
