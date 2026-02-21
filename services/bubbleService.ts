import { BubbleIndexData, BubbleHistoryData, DataPoint, BacktestResults, GSADFResults, MarkovRegimes } from '../types';

const BASE = '/Market-Bubble-Index-Dashboard/';

const NO_CACHE = { cache: 'no-store' as RequestCache };

export const fetchBubbleIndex = async (): Promise<BubbleIndexData> => {
  const res = await fetch(`${BASE}data/bubble_index.json`, NO_CACHE);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
};

export const fetchBubbleHistory = async (): Promise<BubbleHistoryData> => {
  const res = await fetch(`${BASE}data/bubble_history.json`, NO_CACHE);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
};

export const fetchTickerPrice = async (ticker: string): Promise<DataPoint[]> => {
  try {
    const res = await fetch(`${BASE}data/${ticker.toLowerCase()}.json`, NO_CACHE);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json = await res.json();
    return json.data ?? [];
  } catch {
    return [];
  }
};

export const fetchBacktestResults = async (): Promise<BacktestResults | null> => {
  try {
    const res = await fetch(`${BASE}data/backtest_results.json`, NO_CACHE);
    if (!res.ok) return null;
    return res.json();
  } catch { return null; }
};

export const fetchGSADFResults = async (): Promise<GSADFResults | null> => {
  try {
    const res = await fetch(`${BASE}data/gsadf_results.json`, NO_CACHE);
    if (!res.ok) return null;
    return res.json();
  } catch { return null; }
};

export const fetchMarkovRegimes = async (): Promise<MarkovRegimes | null> => {
  try {
    const res = await fetch(`${BASE}data/markov_regimes.json`, NO_CACHE);
    if (!res.ok) return null;
    return res.json();
  } catch { return null; }
};

export interface DrawdownModelData {
  model_version: string;
  calibration_date: string;
  forward_window_days: number;
  forward_window_label: string;
  train_test_split?: string;
  feature_names?: string[];
  current_features?: Record<string, number>;
  logistic_coefficients: Record<string, {
    // v3.0 per-threshold optimized format
    features?: string[];
    weights?: Record<string, number>;
    intercept?: number;
    scaler_mean?: Record<string, number>;
    scaler_std?: Record<string, number>;
    dd_definition?: string;
    forward_window?: number;
    regularization_C?: number;
    n_train?: number;
    n_test?: number;
    n_events_train?: number;
    n_events_test?: number;
    base_rate_train?: number;
    base_rate_test?: number;
    auc_train?: number;
    auc_test?: number;
    brier_train?: number;
    brier_test?: number;
    bss_test?: number;
    // v1.0 legacy format
    a?: number;
    b?: number;
    a_velocity?: number;
    b_with_velocity?: number;
    a_score_with_velocity?: number;
    n_events?: number;
    n_total?: number;
  }>;
  bayesian_lookup: Record<string, {
    bin_centers: number[];
    probabilities: number[];
  }>;
  evt_parameters: {
    gpd_shape_xi: number;
    gpd_scale_sigma: number;
    threshold_u: number;
    n_exceedances: number;
    exceedance_ratios: Record<string, number>;
    cross_ratios: Record<string, number>;
  };
  empirical_stats: Record<string, Record<string, number>>;
  confidence_tiers: Record<string, string>;
}

export const fetchDrawdownModel = async (): Promise<DrawdownModelData | null> => {
  try {
    const res = await fetch(`${BASE}data/drawdown_model.json`, NO_CACHE);
    if (!res.ok) return null;
    return res.json();
  } catch { return null; }
};

export interface DrawdownPoint {
  date: string;
  drawdown: number;
}

export const fetchQQQDrawdown = async (): Promise<DrawdownPoint[]> => {
  try {
    const res = await fetch(`${BASE}data/qqq_drawdown.json`, NO_CACHE);
    if (!res.ok) return [];
    return res.json();
  } catch { return []; }
};
