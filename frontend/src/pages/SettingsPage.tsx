import React, { useState, useEffect } from 'react';
import {
  Settings,
  User,
  Eye,
  Volume2,
  Shield,
  Trash2,
  Check,
  Cpu,
  AlertCircle,
  Save,
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { useAccessibility } from '../context/AccessibilityContext';
import { api } from '../api/client';

export const SettingsPage: React.FC = () => {
  const { user, refreshUser } = useAuth();
  const {
    highContrast,
    toggleHighContrast,
    largeText,
    toggleLargeText,
    reducedMotion,
    toggleReducedMotion,
  } = useAccessibility();

  const [preferredLang, setPreferredLang] = useState<string>('en');
  const [autoSpeak, setAutoSpeak] = useState<boolean>(true);
  const [storeHistory, setStoreHistory] = useState<boolean>(true);
  const [saveStatus, setSaveStatus] = useState<string | null>(null);
  const [wipeStatus, setWipeStatus] = useState<string | null>(null);

  useEffect(() => {
    if (user?.preferences) {
      setPreferredLang(user.preferences.preferred_language || 'en');
      setAutoSpeak(user.preferences.auto_speak ?? true);
      setStoreHistory(user.preferences.store_history ?? true);
    }
  }, [user]);

  const handleSavePreferences = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await api.updatePreferences({
        preferred_language: preferredLang as 'en' | 'hi',
        auto_speak: autoSpeak,
        store_history: storeHistory,
        high_contrast: highContrast,
        reduced_motion: reducedMotion,
        font_size: largeText ? 'large' : 'normal',
      });
      await refreshUser();
      setSaveStatus('Preferences successfully saved!');
      setTimeout(() => setSaveStatus(null), 2500);
    } catch (err: any) {
      setSaveStatus(`Error saving: ${err.message}`);
      setTimeout(() => setSaveStatus(null), 3000);
    }
  };

  const handleWipeHistory = async () => {
    if (!window.confirm('Are you sure you want to delete all stored recognition and practice history? This action cannot be undone.')) {
      return;
    }

    try {
      await api.clearHistory(true);
      setWipeStatus('Your history has been completely wiped from the database.');
      setTimeout(() => setWipeStatus(null), 3000);
    } catch (err: any) {
      setWipeStatus(`Error clearing data: ${err.message}`);
      setTimeout(() => setWipeStatus(null), 3000);
    }
  };

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Header */}
      <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 sm:p-8">
        <div className="flex items-center space-x-2">
          <span className="text-xs font-bold text-teal-400 uppercase tracking-wider">Configuration Hub</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-white mt-1">
          Settings & Accessibility Preferences
        </h1>
        <p className="text-xs text-slate-400 mt-1">
          Customize assistive display parameters, speech synthesis behavior, and privacy retention rules.
        </p>
      </div>

      {saveStatus && (
        <div className="p-3.5 rounded-2xl bg-teal-500/10 border border-teal-500/30 text-teal-300 text-xs flex items-center space-x-2">
          <Check className="w-4 h-4 flex-shrink-0" />
          <span>{saveStatus}</span>
        </div>
      )}

      {wipeStatus && (
        <div className="p-3.5 rounded-2xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-center space-x-2">
          <Check className="w-4 h-4 flex-shrink-0" />
          <span>{wipeStatus}</span>
        </div>
      )}

      {/* User Profile Overview */}
      <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 space-y-4">
        <h2 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
          <User className="w-4 h-4 text-teal-400" /> Account Profile
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
          <div className="p-4 rounded-2xl bg-slate-950 border border-slate-800">
            <span className="text-slate-400 block mb-1">Email Address:</span>
            <span className="text-white font-bold font-mono">{user?.email || 'demo@signbridge.ai'}</span>
          </div>
          <div className="p-4 rounded-2xl bg-slate-950 border border-slate-800">
            <span className="text-slate-400 block mb-1">Full Name:</span>
            <span className="text-white font-bold">{user?.full_name || 'SignBridge Student'}</span>
          </div>
        </div>
      </div>

      {/* Accessibility Controls */}
      <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 space-y-4">
        <h2 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
          <Eye className="w-4 h-4 text-teal-400" /> Accessibility (WCAG 2.1 AA)
        </h2>

        <div className="space-y-3">
          <div className="flex items-center justify-between p-4 rounded-2xl bg-slate-950 border border-slate-800">
            <div>
              <p className="text-sm font-bold text-white">High Contrast Mode</p>
              <p className="text-xs text-slate-400">Pure black background with vivid maximum-contrast outlines</p>
            </div>
            <button
              onClick={toggleHighContrast}
              className={`w-12 h-6 rounded-full transition-colors relative ${
                highContrast ? 'bg-teal-500' : 'bg-slate-800'
              }`}
            >
              <span
                className={`w-5 h-5 rounded-full bg-white absolute top-0.5 transition-transform ${
                  highContrast ? 'left-6.5' : 'left-0.5'
                }`}
              />
            </button>
          </div>

          <div className="flex items-center justify-between p-4 rounded-2xl bg-slate-950 border border-slate-800">
            <div>
              <p className="text-sm font-bold text-white">Large Accessible Text (A+)</p>
              <p className="text-xs text-slate-400">Enlarges UI font scale for enhanced legibility</p>
            </div>
            <button
              onClick={toggleLargeText}
              className={`w-12 h-6 rounded-full transition-colors relative ${
                largeText ? 'bg-teal-500' : 'bg-slate-800'
              }`}
            >
              <span
                className={`w-5 h-5 rounded-full bg-white absolute top-0.5 transition-transform ${
                  largeText ? 'left-6.5' : 'left-0.5'
                }`}
              />
            </button>
          </div>

          <div className="flex items-center justify-between p-4 rounded-2xl bg-slate-950 border border-slate-800">
            <div>
              <p className="text-sm font-bold text-white">Reduced Motion</p>
              <p className="text-xs text-slate-400">Minimizes animations and live UI transitions</p>
            </div>
            <button
              onClick={toggleReducedMotion}
              className={`w-12 h-6 rounded-full transition-colors relative ${
                reducedMotion ? 'bg-teal-500' : 'bg-slate-800'
              }`}
            >
              <span
                className={`w-5 h-5 rounded-full bg-white absolute top-0.5 transition-transform ${
                  reducedMotion ? 'left-6.5' : 'left-0.5'
                }`}
              />
            </button>
          </div>
        </div>
      </div>

      {/* Speech & Language Settings */}
      <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 space-y-4">
        <h2 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
          <Volume2 className="w-4 h-4 text-teal-400" /> Speech & Synthesis Options
        </h2>

        <div className="space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 p-4 rounded-2xl bg-slate-950 border border-slate-800">
            <div>
              <p className="text-sm font-bold text-white">Preferred Output Language</p>
              <p className="text-xs text-slate-400">Default voice and translation target</p>
            </div>
            <select
              value={preferredLang}
              onChange={(e) => setPreferredLang(e.target.value)}
              className="px-3 py-1.5 rounded-xl bg-slate-900 border border-slate-800 text-white text-xs font-semibold focus:outline-none"
            >
              <option value="en">English (US / India)</option>
              <option value="hi">Hindi (Devanagari / Romanized)</option>
            </select>
          </div>

          <div className="flex items-center justify-between p-4 rounded-2xl bg-slate-950 border border-slate-800">
            <div>
              <p className="text-sm font-bold text-white">Auto-Speak Recognized Tokens</p>
              <p className="text-xs text-slate-400">Automatically voices recognized signs aloud as you sign</p>
            </div>
            <input
              type="checkbox"
              checked={autoSpeak}
              onChange={(e) => setAutoSpeak(e.target.checked)}
              className="w-5 h-5 accent-teal-500 rounded cursor-pointer"
            />
          </div>
        </div>
      </div>

      {/* Privacy & History Retention Controls */}
      <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 space-y-4">
        <h2 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
          <Shield className="w-4 h-4 text-emerald-400" /> Privacy & Data Retention
        </h2>

        <div className="space-y-4">
          <div className="flex items-center justify-between p-4 rounded-2xl bg-slate-950 border border-slate-800">
            <div>
              <p className="text-sm font-bold text-white">Store Recognition History</p>
              <p className="text-xs text-slate-400">Save detected signs to your account for learning analytics</p>
            </div>
            <input
              type="checkbox"
              checked={storeHistory}
              onChange={(e) => setStoreHistory(e.target.checked)}
              className="w-5 h-5 accent-teal-500 rounded cursor-pointer"
            />
          </div>

          {/* Privacy Erasure */}
          <div className="p-4 rounded-2xl bg-rose-950/20 border border-rose-500/30 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
            <div>
              <p className="text-sm font-bold text-rose-300">Wipe All Telemetry Data (GDPR Right to be Forgotten)</p>
              <p className="text-xs text-slate-400">
                Permanently purge all saved recognition events and practice records from the database.
              </p>
            </div>
            <button
              onClick={handleWipeHistory}
              className="px-4 py-2 rounded-xl bg-rose-600 hover:bg-rose-500 text-white font-bold text-xs flex items-center space-x-1.5 transition-colors whitespace-nowrap"
            >
              <Trash2 className="w-4 h-4" />
              <span>Wipe Data</span>
            </button>
          </div>
        </div>
      </div>

      {/* Save Settings Trigger */}
      <div className="flex justify-end">
        <button
          onClick={handleSavePreferences}
          className="px-6 py-3 rounded-xl bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold text-sm flex items-center space-x-2 shadow-lg shadow-teal-500/20"
        >
          <Save className="w-4 h-4" />
          <span>Save Preferences</span>
        </button>
      </div>
    </div>
  );
};
