import { BubbleIndexData, BubbleHistoryData, DataPoint, BacktestResults, GSADFResults, MarkovRegimes } from '../types';

const BASE = '/Market-Bubble-Index-Dashboard/';

export const fetchBubbleIndex = async (): Promise<BubbleIndexData> => {
  const res = await fetch(`${BASE}data/bubble_index.json`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
};

export const fetchBubbleHistory = async (): Promise<BubbleHistoryData> => {
  const res = await fetch(`${BASE}data/bubble_history.json`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
};

export const fetchTickerPrice = async (ticker: string): Promise<DataPoint[]> => {
  try {
    const res = await fetch(`${BASE}data/${ticker.toLowerCase()}.json`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json = await res.json();
    return json.data ?? [];
  } catch {
    return [];
  }
};

export const fetchBacktestResults = async (): Promise<BacktestResults | null> => {
  try {
    const res = await fetch(`${BASE}data/backtest_results.json`);
    if (!res.ok) return null;
    return res.json();
  } catch { return null; }
};

export const fetchGSADFResults = async (): Promise<GSADFResults | null> => {
  try {
    const res = await fetch(`${BASE}data/gsadf_results.json`);
    if (!res.ok) return null;
    return res.json();
  } catch { return null; }
};

export const fetchMarkovRegimes = async (): Promise<MarkovRegimes | null> => {
  try {
    const res = await fetch(`${BASE}data/markov_regimes.json`);
    if (!res.ok) return null;
    return res.json();
  } catch { return null; }
};
