import React from 'react';
import type { TabId } from '../constants';

interface Tab {
  id: TabId;
  label: string;
  icon: string;
  accentColor: string;
}

const TABS: Tab[] = [
  { id: 'bubble', label: 'Bubble Temp', icon: '🌡', accentColor: '#3b82f6' },
  { id: 'risk', label: 'Crash Risk', icon: '⚠', accentColor: '#ef4444' },
  { id: 'deviation', label: 'Deviation', icon: '📊', accentColor: '#3b82f6' },
];

interface TabSelectorProps {
  activeTab: TabId;
  onTabChange: (tab: TabId) => void;
}

const TabSelector: React.FC<TabSelectorProps> = ({ activeTab, onTabChange }) => {
  return (
    <div className="flex items-center bg-slate-800 rounded-lg p-1 gap-0.5">
      {TABS.map((tab) => {
        const isActive = activeTab === tab.id;
        return (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className={`px-3 py-1.5 rounded-md text-sm font-semibold transition-colors whitespace-nowrap flex items-center gap-1.5 ${
              isActive
                ? 'text-white shadow-sm'
                : 'text-slate-400 hover:text-white hover:bg-slate-700'
            }`}
            style={isActive ? { backgroundColor: tab.accentColor } : undefined}
          >
            <span className="hidden sm:inline">{tab.icon}</span>
            <span className="hidden sm:inline">{tab.label}</span>
            <span className="sm:hidden">{tab.icon}</span>
          </button>
        );
      })}
    </div>
  );
};

export default TabSelector;
