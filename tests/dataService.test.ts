import { describe, it, expect } from "vitest";
import { generateHistoricalData, getMarketSummary } from "../services/dataService";

describe("generateHistoricalData", () => {
  it("should generate data points for the requested years", () => {
    const data = generateHistoricalData(1);
    // ~252 trading days per year + 1
    expect(data.length).toBeGreaterThan(200);
    expect(data.length).toBeLessThan(300);
  });

  it("should produce valid data points", () => {
    const data = generateHistoricalData(1);
    for (const point of data) {
      expect(point.price).toBeGreaterThan(0);
      expect(point.sma200).toBeGreaterThan(0);
      expect(point.index).toBeGreaterThanOrEqual(0);
      expect(point.index).toBeLessThanOrEqual(100);
      expect(point.date).toBeDefined();
    }
  });
});

describe("getMarketSummary", () => {
  it("should return a valid summary", () => {
    const data = generateHistoricalData(1);
    const summary = getMarketSummary(data);
    expect(summary.currentPrice).toBeGreaterThan(0);
    expect(summary.currentSMA).toBeGreaterThan(0);
    expect(summary.currentIndex).toBeGreaterThanOrEqual(0);
    expect(summary.currentIndex).toBeLessThanOrEqual(100);
    expect(["Low", "Moderate", "High", "Danger"]).toContain(summary.riskLevel);
  });
});
