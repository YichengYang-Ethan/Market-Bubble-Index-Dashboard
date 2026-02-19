import React from 'react';
import { BubbleIndicator, BubbleHistoryPoint, PreviousDay } from '../types';
import { INDICATOR_META } from '../constants';
import IndicatorCard from './IndicatorCard';

interface IndicatorGridProps {
  indicators: Record<string, BubbleIndicator>;
  previousDay?: PreviousDay;
  history: BubbleHistoryPoint[];
  perspective?: 'bubble' | 'risk';
}

const IndicatorGrid: React.FC<IndicatorGridProps> = ({ indicators, previousDay, history, perspective = 'bubble' }) => {
  const recentHistory = history.slice(-30);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {INDICATOR_META.map((meta) => {
        const indicator = indicators[meta.key];
        if (!indicator) return null;

        const sparklineData: (number | null)[] = recentHistory.map(
          (point) => point.indicators?.[meta.key] ?? null
        );

        const previousScore = previousDay?.indicators?.[meta.key] ?? null;

        return (
          <IndicatorCard
            key={meta.key}
            indicatorKey={meta.key}
            indicator={indicator}
            previousScore={previousScore}
            sparklineData={sparklineData}
            perspective={perspective}
          />
        );
      })}
    </div>
  );
};

export default IndicatorGrid;
