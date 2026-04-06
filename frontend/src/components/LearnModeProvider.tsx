import { createContext, useContext, useCallback, useEffect, useState, type ReactNode } from 'react';
import { TOUR_SEQUENCE } from './learnContent';

interface LearnModeState {
  active: boolean;
  currentPane: string | null;
  mode: 'free' | 'guided';
  visitedPanes: Set<string>;
  tourIndex: number;
}

interface LearnModeContextValue extends LearnModeState {
  enterLearnMode: () => void;
  exitLearnMode: () => void;
  setActivePane: (id: string) => void;
  startTour: () => void;
  nextPane: () => void;
  prevPane: () => void;
}

const LearnModeContext = createContext<LearnModeContextValue | null>(null);

export function useLearnMode() {
  const ctx = useContext(LearnModeContext);
  if (!ctx) throw new Error('useLearnMode must be used within LearnModeProvider');
  return ctx;
}

const INITIAL_STATE: LearnModeState = {
  active: false,
  currentPane: null,
  mode: 'free',
  visitedPanes: new Set(),
  tourIndex: 0,
};

export function LearnModeProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<LearnModeState>(INITIAL_STATE);

  const enterLearnMode = useCallback(() => {
    setState({
      active: true,
      currentPane: TOUR_SEQUENCE[0],
      mode: 'free',
      visitedPanes: new Set([TOUR_SEQUENCE[0]]),
      tourIndex: 0,
    });
  }, []);

  const exitLearnMode = useCallback(() => {
    setState(INITIAL_STATE);
  }, []);

  const setActivePane = useCallback((id: string) => {
    setState(prev => ({
      ...prev,
      currentPane: id,
      mode: 'free',
      visitedPanes: new Set([...prev.visitedPanes, id]),
    }));
  }, []);

  const startTour = useCallback(() => {
    setState(prev => ({
      ...prev,
      mode: 'guided',
      tourIndex: 0,
      currentPane: TOUR_SEQUENCE[0],
      visitedPanes: new Set([...prev.visitedPanes, TOUR_SEQUENCE[0]]),
    }));
  }, []);

  const nextPane = useCallback(() => {
    setState(prev => {
      if (prev.mode !== 'guided') return prev;
      const nextIdx = prev.tourIndex + 1;
      if (nextIdx >= TOUR_SEQUENCE.length) {
        return INITIAL_STATE;
      }
      const nextId = TOUR_SEQUENCE[nextIdx];
      return {
        ...prev,
        tourIndex: nextIdx,
        currentPane: nextId,
        visitedPanes: new Set([...prev.visitedPanes, nextId]),
      };
    });
  }, []);

  const prevPane = useCallback(() => {
    setState(prev => {
      if (prev.mode !== 'guided' || prev.tourIndex <= 0) return prev;
      const prevIdx = prev.tourIndex - 1;
      return {
        ...prev,
        tourIndex: prevIdx,
        currentPane: TOUR_SEQUENCE[prevIdx],
      };
    });
  }, []);

  useEffect(() => {
    if (!state.active) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        exitLearnMode();
      } else if (state.mode === 'guided') {
        if (e.key === 'ArrowRight') nextPane();
        if (e.key === 'ArrowLeft') prevPane();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [state.active, state.mode, exitLearnMode, nextPane, prevPane]);

  const value: LearnModeContextValue = {
    ...state,
    enterLearnMode,
    exitLearnMode,
    setActivePane,
    startTour,
    nextPane,
    prevPane,
  };

  return (
    <LearnModeContext.Provider value={value}>
      {children}
    </LearnModeContext.Provider>
  );
}
