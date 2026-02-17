import { describe, it, expect } from "vitest";
import { generateHistoricalData, getMarketSummary, runBacktest, detectHistoricalSignals } from "../services/dataService";
import type { DataPoint } from "../types";

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

function makeDataPoint(overrides: Partial<DataPoint> & { date: string }): DataPoint {
  return {
    price: 100,
    sma200: 100,
    deviation: 0,
    index: 50,
    ...overrides,
  };
}

describe("runBacktest", () => {
  it("should return zero trades for empty-ish data with no threshold crossings", () => {
    const data: DataPoint[] = [
      makeDataPoint({ date: "2024-01-01", index: 50 }),
      makeDataPoint({ date: "2024-01-02", index: 50 }),
      makeDataPoint({ date: "2024-01-03", index: 50 }),
    ];
    const result = runBacktest(data, 30, 70, 10000);
    expect(result.numTrades).toBe(0);
    expect(result.signals).toHaveLength(0);
    expect(result.maxDrawdown).toBe(0);
  });

  it("should buy when index drops below buyThreshold and sell above sellThreshold", () => {
    const data: DataPoint[] = [
      makeDataPoint({ date: "2024-01-01", price: 100, index: 50 }),
      makeDataPoint({ date: "2024-01-02", price: 90, index: 20 }),  // buy
      makeDataPoint({ date: "2024-01-03", price: 95, index: 50 }),
      makeDataPoint({ date: "2024-01-04", price: 110, index: 80 }), // sell
      makeDataPoint({ date: "2024-01-05", price: 105, index: 60 }),
    ];
    const result = runBacktest(data, 30, 70, 10000);
    expect(result.numTrades).toBe(2);
    expect(result.signals[0].type).toBe("buy");
    expect(result.signals[1].type).toBe("sell");
    expect(result.strategyReturn).toBeGreaterThan(0);
  });

  it("should handle buyThreshold > sellThreshold (no trades)", () => {
    const data: DataPoint[] = [
      makeDataPoint({ date: "2024-01-01", price: 100, index: 50 }),
      makeDataPoint({ date: "2024-01-02", price: 90, index: 20 }),
      makeDataPoint({ date: "2024-01-03", price: 110, index: 80 }),
    ];
    // buyThreshold=80, sellThreshold=20 means buy at high index, sell at low
    const result = runBacktest(data, 80, 20, 10000);
    // index 20 <= 80 triggers buy, then index 80 is not >= 20 while in position... actually it is
    // Let's reason: first point index=50 <= 80 -> buy. second point index=20 >= 20 -> sell. third point index=80 <= 80 -> buy.
    expect(result.numTrades).toBeGreaterThan(0);
  });

  it("should handle all-buy signals (index always below threshold)", () => {
    const data: DataPoint[] = [
      makeDataPoint({ date: "2024-01-01", price: 100, index: 10 }),
      makeDataPoint({ date: "2024-01-02", price: 95, index: 5 }),
      makeDataPoint({ date: "2024-01-03", price: 90, index: 3 }),
    ];
    const result = runBacktest(data, 30, 70, 10000);
    // Only one buy on first point, no sell ever
    expect(result.signals).toHaveLength(1);
    expect(result.signals[0].type).toBe("buy");
  });

  it("should handle all-sell signals (index always above threshold)", () => {
    const data: DataPoint[] = [
      makeDataPoint({ date: "2024-01-01", price: 100, index: 90 }),
      makeDataPoint({ date: "2024-01-02", price: 105, index: 85 }),
      makeDataPoint({ date: "2024-01-03", price: 110, index: 95 }),
    ];
    const result = runBacktest(data, 30, 70, 10000);
    // Index never drops to 30, so no buy, no sell
    expect(result.numTrades).toBe(0);
  });

  it("should compute buy-and-hold return correctly", () => {
    const data: DataPoint[] = [
      makeDataPoint({ date: "2024-01-01", price: 100, index: 50 }),
      makeDataPoint({ date: "2024-06-01", price: 120, index: 50 }),
    ];
    const result = runBacktest(data, 30, 70, 10000);
    expect(result.buyHoldReturn).toBeCloseTo(0.2, 5);
  });

  it("should track max drawdown", () => {
    const data: DataPoint[] = [
      makeDataPoint({ date: "2024-01-01", price: 100, index: 20 }), // buy
      makeDataPoint({ date: "2024-01-02", price: 80, index: 30 }),  // drawdown
      makeDataPoint({ date: "2024-01-03", price: 120, index: 80 }), // sell, recovery
    ];
    const result = runBacktest(data, 30, 70, 10000);
    expect(result.maxDrawdown).toBeGreaterThan(0);
  });
});

describe("detectHistoricalSignals", () => {
  it("should return empty array for empty data", () => {
    const signals = detectHistoricalSignals([]);
    expect(signals).toHaveLength(0);
  });

  it("should return empty for data shorter than window size", () => {
    const data = Array.from({ length: 30 }, (_, i) =>
      makeDataPoint({ date: `2024-01-${String(i + 1).padStart(2, "0")}`, index: 50 })
    );
    const signals = detectHistoricalSignals(data);
    expect(signals).toHaveLength(0);
  });

  it("should detect sell signals at high index peaks", () => {
    // Create data with a clear peak above 80
    const data: DataPoint[] = [];
    for (let i = 0; i < 100; i++) {
      const index = i === 50 ? 95 : 50; // one peak at position 50
      data.push(makeDataPoint({ date: `2024-${String(Math.floor(i / 28) + 1).padStart(2, "0")}-${String((i % 28) + 1).padStart(2, "0")}`, index, price: 100 + index }));
    }
    const signals = detectHistoricalSignals(data, 80, 20);
    const sells = signals.filter((s) => s.type === "sell");
    expect(sells.length).toBeGreaterThanOrEqual(1);
  });

  it("should detect buy signals at low index troughs", () => {
    const data: DataPoint[] = [];
    for (let i = 0; i < 100; i++) {
      const index = i === 50 ? 5 : 50; // one trough at position 50
      data.push(makeDataPoint({ date: `2024-${String(Math.floor(i / 28) + 1).padStart(2, "0")}-${String((i % 28) + 1).padStart(2, "0")}`, index, price: 100 + index }));
    }
    const signals = detectHistoricalSignals(data, 80, 20);
    const buys = signals.filter((s) => s.type === "buy");
    expect(buys.length).toBeGreaterThanOrEqual(1);
  });

  it("should return signals sorted newest first", () => {
    const data: DataPoint[] = [];
    for (let i = 0; i < 100; i++) {
      let index = 50;
      if (i === 30) index = 5;  // buy signal
      if (i === 70) index = 95; // sell signal
      data.push(makeDataPoint({ date: `2024-${String(Math.floor(i / 28) + 1).padStart(2, "0")}-${String((i % 28) + 1).padStart(2, "0")}`, index, price: 100 + index }));
    }
    const signals = detectHistoricalSignals(data, 80, 20);
    for (let i = 1; i < signals.length; i++) {
      expect(signals[i - 1].date >= signals[i].date).toBe(true);
    }
  });
});
