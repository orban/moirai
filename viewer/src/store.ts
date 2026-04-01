import { create } from "zustand";

interface ViewerState {
  activeTaskIndex: number;
  setActiveTask: (index: number) => void;
  focusPath: string[];
  drillDown: (branchId: string) => void;
  drillUp: (toIndex: number) => void;
  resetFocus: () => void;
  expandedRuns: Set<string>;
  toggleRun: (runId: string) => void;
}

export const useViewerStore = create<ViewerState>((set) => ({
  activeTaskIndex: 0,
  setActiveTask: (index) => set({ activeTaskIndex: index, focusPath: [], expandedRuns: new Set() }),
  focusPath: [],
  drillDown: (branchId) => set((s) => ({ focusPath: [...s.focusPath, branchId] })),
  drillUp: (toIndex) => set((s) => ({ focusPath: s.focusPath.slice(0, toIndex + 1) })),
  resetFocus: () => set({ focusPath: [] }),
  expandedRuns: new Set(),
  toggleRun: (runId) =>
    set((s) => {
      const next = new Set(s.expandedRuns);
      if (next.has(runId)) next.delete(runId);
      else next.add(runId);
      return { expandedRuns: next };
    }),
}));
