import React, { createContext, useContext, useEffect, useState } from 'react';

interface AccessibilityContextType {
  highContrast: boolean;
  toggleHighContrast: () => void;
  largeText: boolean;
  toggleLargeText: () => void;
  reducedMotion: boolean;
  toggleReducedMotion: () => void;
  speakText: (text: string, lang?: string) => void;
  stopSpeaking: () => void;
  isSpeaking: boolean;
  theme: 'dark' | 'light';
  toggleTheme: () => void;
}

const AccessibilityContext = createContext<AccessibilityContextType | undefined>(undefined);

export const AccessibilityProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [highContrast, setHighContrast] = useState<boolean>(() => {
    return localStorage.getItem('sb_high_contrast') === 'true';
  });

  const [largeText, setLargeText] = useState<boolean>(() => {
    return localStorage.getItem('sb_large_text') === 'true';
  });

  const [reducedMotion, setReducedMotion] = useState<boolean>(() => {
    return localStorage.getItem('sb_reduced_motion') === 'true';
  });

  const [theme, setTheme] = useState<'dark' | 'light'>('dark');
  const [isSpeaking, setIsSpeaking] = useState<boolean>(false);

  useEffect(() => {
    const root = document.documentElement;
    const body = document.body;

    if (highContrast) {
      body.classList.add('high-contrast');
    } else {
      body.classList.remove('high-contrast');
    }

    if (largeText) {
      body.classList.add('large-text');
    } else {
      body.classList.remove('large-text');
    }

    if (reducedMotion) {
      body.classList.add('reduced-motion');
    } else {
      body.classList.remove('reduced-motion');
    }

    if (theme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
  }, [highContrast, largeText, reducedMotion, theme]);

  const toggleHighContrast = () => {
    setHighContrast((prev) => {
      const next = !prev;
      localStorage.setItem('sb_high_contrast', String(next));
      return next;
    });
  };

  const toggleLargeText = () => {
    setLargeText((prev) => {
      const next = !prev;
      localStorage.setItem('sb_large_text', String(next));
      return next;
    });
  };

  const toggleReducedMotion = () => {
    setReducedMotion((prev) => {
      const next = !prev;
      localStorage.setItem('sb_reduced_motion', String(next));
      return next;
    });
  };

  const toggleTheme = () => {
    setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'));
  };

  const speakText = (text: string, lang: string = 'en') => {
    if (!text || typeof window === 'undefined' || !('speechSynthesis' in window)) {
      return;
    }
    try {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = lang === 'hi' ? 'hi-IN' : 'en-US';
      utterance.rate = 0.95;
      utterance.pitch = 1.0;

      utterance.onstart = () => setIsSpeaking(true);
      utterance.onend = () => setIsSpeaking(false);
      utterance.onerror = () => setIsSpeaking(false);

      window.speechSynthesis.speak(utterance);
    } catch (e) {
      console.warn('Speech synthesis error:', e);
      setIsSpeaking(false);
    }
  };

  const stopSpeaking = () => {
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      setIsSpeaking(false);
    }
  };

  return (
    <AccessibilityContext.Provider
      value={{
        highContrast,
        toggleHighContrast,
        largeText,
        toggleLargeText,
        reducedMotion,
        toggleReducedMotion,
        speakText,
        stopSpeaking,
        isSpeaking,
        theme,
        toggleTheme,
      }}
    >
      {children}
    </AccessibilityContext.Provider>
  );
};

export const useAccessibility = () => {
  const context = useContext(AccessibilityContext);
  if (!context) {
    throw new Error('useAccessibility must be used within an AccessibilityProvider');
  }
  return context;
};
