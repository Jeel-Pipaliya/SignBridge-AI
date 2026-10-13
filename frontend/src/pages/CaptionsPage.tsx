import React, { useState, useEffect, useRef } from 'react';
import {
  Captions,
  Mic,
  MicOff,
  Download,
  Trash2,
  Shield,
  Clock,
  User,
  Settings2,
  Copy,
  Check,
  Info,
} from 'lucide-react';
import { api } from '../api/client';

interface CaptionEntry {
  id: string;
  speaker: string;
  text: string;
  timestamp: string;
}

export const CaptionsPage: React.FC = () => {
  const [isCapturing, setIsCapturing] = useState<boolean>(false);
  const [activeSpeaker, setActiveSpeaker] = useState<string>('Speaker 1');
  const [entries, setEntries] = useState<CaptionEntry[]>([
    {
      id: '1',
      speaker: 'Lecturer',
      text: 'Good morning everyone. Today we are reviewing computer vision pipelines in Indian Sign Language translation.',
      timestamp: '09:30:15',
    },
    {
      id: '2',
      speaker: 'Student (Question)',
      text: 'Could you explain why translation invariance requires wrist centering?',
      timestamp: '09:30:42',
    },
    {
      id: '3',
      speaker: 'Lecturer',
      text: 'Because shifting the origin to landmark zero ensures the coordinate vector is identical regardless of where the hand appears on camera.',
      timestamp: '09:31:05',
    },
  ]);

  const [currentInterim, setCurrentInterim] = useState<string>('');
  const [copied, setCopied] = useState<boolean>(false);
  const [consentAcknowledged, setConsentAcknowledged] = useState<boolean>(true);
  const [isSimulating, setIsSimulating] = useState<boolean>(false);
  const [manualInput, setManualInput] = useState<string>('');
  const [micNotice, setMicNotice] = useState<string | null>(null);

  const isCapturingRef = useRef<boolean>(false);
  const simulationTimerRef = useRef<any>(null);

  const SIMULATED_LECTURE_STREAM = [
    { speaker: 'Professor Rao', text: 'Welcome everyone. In today\'s session, we examine Indian Sign Language grammar reconstruction.' },
    { speaker: 'Student (Ananya)', text: 'Professor, how does the system preserve sentence structure with Subject-Object-Verb ordering?' },
    { speaker: 'Professor Rao', text: 'ISL adheres to SOV order, whereas standard English utilizes Subject-Verb-Object. The deterministic grammar engine aligns the parts of speech.' },
    { speaker: 'Guest Lecturer', text: 'Moreover, real-time MediaPipe feature normalization guarantees that wrist translation does not degrade landmark predictions.' },
    { speaker: 'Student (Rohan)', text: 'Is the offline Text-to-Speech service accessible in multiple Indian languages?' },
    { speaker: 'Professor Rao', text: 'Yes, SignBridge AI includes both English phonetics and Hindi Devanagari transliteration with bilingual audio synthesis.' },
  ];

  const startCaptions = () => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setMicNotice('Web Speech API is unavailable in this browser. You can click "Simulate Meeting Stream" or type live captions below.');
      return;
    }

    try {
      setMicNotice(null);
      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = 'en-US';

      recognition.onstart = () => {
        setIsCapturing(true);
        isCapturingRef.current = true;
      };

      recognition.onresult = (event: any) => {
        let interim = '';
        for (let i = event.resultIndex; i < event.results.length; ++i) {
          if (event.results[i].isFinal) {
            const finalTxt = event.results[i][0].transcript.trim();
            if (finalTxt) {
              const newEntry: CaptionEntry = {
                id: String(Date.now()),
                speaker: activeSpeaker,
                text: finalTxt,
                timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
              };
              setEntries((prev) => [...prev, newEntry]);
            }
          } else {
            interim += event.results[i][0].transcript;
          }
        }
        setCurrentInterim(interim);
      };

      recognition.onerror = (e: any) => {
        console.warn('Speech caption notice:', e);
        if (e.error === 'not-allowed') {
          setMicNotice('Microphone access was blocked. Please permit microphone permissions in your browser or run the Simulation mode.');
        }
      };

      recognition.onend = () => {
        // Auto-restart if user has not explicitly stopped
        if (isCapturingRef.current) {
          try {
            recognition.start();
          } catch {
            setIsCapturing(false);
            isCapturingRef.current = false;
          }
        } else {
          setIsCapturing(false);
          setCurrentInterim('');
        }
      };

      (window as any)._activeCaptionRecognition = recognition;
      recognition.start();
    } catch (err: any) {
      setMicNotice(`Could not initialize microphone: ${err.message}. Try "Simulate Meeting Stream" below.`);
    }
  };

  const stopCaptions = () => {
    isCapturingRef.current = false;
    const rec = (window as any)._activeCaptionRecognition;
    if (rec) {
      try {
        rec.stop();
      } catch {}
      (window as any)._activeCaptionRecognition = null;
    }
    setIsCapturing(false);
    setCurrentInterim('');
  };

  // Toggle Simulated Meeting Stream
  const toggleSimulation = () => {
    if (isSimulating) {
      clearInterval(simulationTimerRef.current);
      setIsSimulating(false);
      setCurrentInterim('');
      return;
    }

    setIsSimulating(true);
    let step = 0;

    simulationTimerRef.current = setInterval(() => {
      if (step < SIMULATED_LECTURE_STREAM.length) {
        const item = SIMULATED_LECTURE_STREAM[step];
        const newEntry: CaptionEntry = {
          id: String(Date.now()),
          speaker: item.speaker,
          text: item.text,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
        };
        setEntries((prev) => [...prev, newEntry]);
        step += 1;
      } else {
        clearInterval(simulationTimerRef.current);
        setIsSimulating(false);
      }
    }, 3500);
  };

  useEffect(() => {
    return () => {
      if (simulationTimerRef.current) {
        clearInterval(simulationTimerRef.current);
      }
    };
  }, []);

  const handlePostManualCaption = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!manualInput.trim()) return;

    const newEntry: CaptionEntry = {
      id: String(Date.now()),
      speaker: activeSpeaker,
      text: manualInput.trim(),
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
    };

    setEntries((prev) => [...prev, newEntry]);
    setManualInput('');
  };

  const handleDownload = (format: 'txt' | 'json' | 'srt') => {
    let content = '';
    let filename = `captions-${new Date().toISOString().slice(0, 10)}`;

    if (format === 'json') {
      content = JSON.stringify(entries, null, 2);
      filename += '.json';
    } else if (format === 'srt') {
      content = entries
        .map((e, idx) => `${idx + 1}\n${e.timestamp},000 --> ${e.timestamp},999\n[${e.speaker}] ${e.text}\n`)
        .join('\n');
      filename += '.srt';
    } else {
      content = entries.map((e) => `[${e.timestamp}] ${e.speaker}: ${e.text}`).join('\n');
      filename += '.txt';
    }

    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      {/* Header and Consent Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 bg-slate-900 border border-slate-800 rounded-3xl p-6">
        <div>
          <div className="flex items-center space-x-2">
            <span className="text-xs font-bold text-teal-400 uppercase tracking-wider">Accessibility Studio</span>
            <span className="text-xs px-2 py-0.5 rounded-full bg-slate-800 text-slate-400 font-mono">
              Live Speech Captioning
            </span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white mt-1">
            Meeting & Lecture Captions
          </h1>
          <p className="text-xs text-slate-400 mt-1 max-w-xl">
            Real-time speech-to-text transcript feed with customizable speaker attribution and export options.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={toggleSimulation}
            className={`px-4 py-2.5 rounded-xl text-xs font-semibold flex items-center space-x-2 border transition-colors ${
              isSimulating
                ? 'bg-amber-500/20 text-amber-300 border-amber-500/40 animate-pulse'
                : 'bg-slate-800 hover:bg-slate-700 text-slate-300 border-slate-700'
            }`}
          >
            <span>{isSimulating ? 'Stop Stream Simulation' : 'Simulate Meeting Stream'}</span>
          </button>

          {isCapturing ? (
            <button
              onClick={stopCaptions}
              className="px-4 py-2.5 rounded-xl bg-rose-600 hover:bg-rose-500 text-white font-bold text-xs flex items-center space-x-2 shadow-lg shadow-rose-600/25 animate-pulse"
            >
              <MicOff className="w-4 h-4" />
              <span>Stop Captions</span>
            </button>
          ) : (
            <button
              onClick={startCaptions}
              className="px-5 py-2.5 rounded-xl bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold text-xs flex items-center space-x-2 shadow-lg shadow-teal-500/20"
            >
              <Mic className="w-4 h-4 fill-slate-950" />
              <span>Start Live Captions</span>
            </button>
          )}
        </div>
      </div>

      {micNotice && (
        <div className="p-3.5 rounded-2xl bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs flex items-center space-x-2">
          <Info className="w-4 h-4 flex-shrink-0" />
          <span>{micNotice}</span>
        </div>
      )}


      {/* Recording Consent & Ethics Notice */}
      <div className="p-4 rounded-2xl bg-slate-900/90 border border-slate-800 text-xs text-slate-400 flex items-start justify-between gap-4">
        <div className="flex items-start space-x-3">
          <Shield className="w-4 h-4 text-emerald-400 flex-shrink-0 mt-0.5" />
          <div className="space-y-1">
            <p className="text-white font-semibold">Consent & Recording Transparency Indicator</p>
            <p className="leading-relaxed">
              When microphone capture is activated, an explicit visual indicator is displayed to all meeting participants.
              Transcripts are held in client memory and only saved when exported or committed by the user.
            </p>
          </div>
        </div>
        <div className="flex items-center space-x-2 flex-shrink-0">
          <span className={`w-2.5 h-2.5 rounded-full ${isCapturing ? 'bg-rose-500 animate-ping' : 'bg-slate-600'}`}></span>
          <span className="font-mono text-[11px] text-slate-300">
            {isCapturing ? 'MIC RECORDING' : 'MIC STANDBY'}
          </span>
        </div>
      </div>

      {/* Main Live Stream Container */}
      <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-xl space-y-4">
        {/* Active Speaker Config & Controls */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-4">
          <div className="flex items-center space-x-2 text-xs">
            <User className="w-4 h-4 text-teal-400" />
            <span className="text-slate-400">Current Speaker Label:</span>
            <input
              type="text"
              value={activeSpeaker}
              onChange={(e) => setActiveSpeaker(e.target.value)}
              className="px-2.5 py-1 rounded-lg bg-slate-950 border border-slate-800 text-white text-xs font-semibold focus:outline-none focus:ring-1 focus:ring-teal-500"
            />
          </div>

          <div className="flex items-center gap-1.5 text-xs">
            <span className="text-slate-500 mr-1">Download:</span>
            <button
              onClick={() => handleDownload('txt')}
              className="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 font-mono text-[11px]"
            >
              .TXT
            </button>
            <button
              onClick={() => handleDownload('srt')}
              className="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 font-mono text-[11px]"
            >
              .SRT
            </button>
            <button
              onClick={() => handleDownload('json')}
              className="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 font-mono text-[11px]"
            >
              .JSON
            </button>
            <button
              onClick={() => setEntries([])}
              className="p-1 rounded-lg text-slate-400 hover:text-rose-400 ml-2"
              title="Clear Captions"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Caption Feed Stream */}
        <div className="min-h-[360px] max-h-[460px] overflow-y-auto space-y-3 pr-2">
          {entries.map((entry) => (
            <div
              key={entry.id}
              className="p-4 rounded-2xl bg-slate-950 border border-slate-800/80 space-y-1 hover:border-slate-700 transition-colors"
            >
              <div className="flex items-center justify-between text-xs">
                <span className="font-bold text-teal-400">{entry.speaker}</span>
                <span className="font-mono text-[11px] text-slate-500">{entry.timestamp}</span>
              </div>
              <p className="text-sm text-slate-100 leading-relaxed font-sans">{entry.text}</p>
            </div>
          ))}

          {/* Real-time Interim Streaming Bubble */}
          {currentInterim && (
            <div className="p-4 rounded-2xl bg-teal-950/20 border border-teal-500/30 space-y-1 animate-pulse">
              <span className="text-xs font-bold text-teal-300">{activeSpeaker} (Streaming...)</span>
              <p className="text-sm text-teal-200 italic">{currentInterim}</p>
            </div>
          )}
        </div>

        {/* Live Caption Input / Stenographer Bar */}
        <form onSubmit={handlePostManualCaption} className="pt-4 border-t border-slate-800 flex items-center gap-2">
          <input
            type="text"
            placeholder={`Type real-time caption as ${activeSpeaker}... (Press Enter to post)`}
            value={manualInput}
            onChange={(e) => setManualInput(e.target.value)}
            className="flex-1 px-4 py-2.5 rounded-xl bg-slate-950 border border-slate-800 text-white placeholder-slate-500 text-xs sm:text-sm focus:outline-none focus:ring-2 focus:ring-teal-500"
          />
          <button
            type="submit"
            disabled={!manualInput.trim()}
            className="px-4 py-2.5 rounded-xl bg-teal-500 hover:bg-teal-400 disabled:opacity-40 text-slate-950 font-bold text-xs whitespace-nowrap shadow-md shadow-teal-500/20"
          >
            Post Caption
          </button>
        </form>
      </div>
    </div>

  );
};
