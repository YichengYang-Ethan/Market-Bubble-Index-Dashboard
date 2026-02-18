import { BubbleIndexData, BubbleHistoryData } from '../types';

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
