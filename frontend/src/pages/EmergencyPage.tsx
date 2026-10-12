import React, { useState } from 'react';
import {
  AlertTriangle,
  Volume2,
  Copy,
  Check,
  PhoneCall,
  ShieldAlert,
  Flame,
  HeartPulse,
  UserX,
  MapPin,
  Ambulance,
  LifeBuoy,
} from 'lucide-react';
import { useAccessibility } from '../context/AccessibilityContext';

export const EmergencyPage: React.FC = () => {
  const { speakText } = useAccessibility();

  const [activeMessage, setActiveMessage] = useState<string>(
    'Emergency: I am Deaf and cannot speak. I need urgent medical assistance.'
  );
  const [copied, setCopied] = useState<boolean>(false);

  const emergencyPresets = [
    {
      title: 'Medical Assistance',
      icon: HeartPulse,
      text: 'Medical Emergency: I need a doctor or ambulance immediately.',
      color: 'rose',
    },
    {
      title: 'Call Ambulance',
      icon: Ambulance,
      text: 'Please call an ambulance to my current location.',
      color: 'rose',
    },
    {
      title: 'Deaf / Speech Impaired',
      icon: UserX,
      text: 'I am Deaf and communicate using text and Indian Sign Language. Please write down instructions.',
      color: 'amber',
    },
    {
      title: 'Fire Hazard',
      icon: Flame,
      text: 'Fire Alert: There is an active fire or smoke emergency nearby.',
      color: 'orange',
    },
    {
      title: 'Police Assistance',
      icon: ShieldAlert,
      text: 'I require immediate police assistance. Please help me contact local authorities.',
      color: 'blue',
    },
    {
      title: 'Lost / Need Directions',
      icon: MapPin,
      text: 'I am lost and need assistance finding the nearest hospital or transit station.',
      color: 'teal',
    },
  ];

  const handleSpeakAloud = () => {
    if (activeMessage) {
      speakText(activeMessage, 'en');
    }
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(activeMessage);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Official Emergency Disclaimer */}
      <div className="p-4 sm:p-5 rounded-3xl bg-rose-950/40 border border-rose-500/40 text-rose-200 text-xs sm:text-sm flex items-start space-x-3 shadow-lg shadow-rose-950/30">
        <AlertTriangle className="w-6 h-6 text-rose-400 flex-shrink-0 mt-0.5" />
        <div className="space-y-1">
          <p className="font-extrabold text-white text-base">EMERGENCY ADVISORY (INDIA DIAL 112)</p>
          <p className="leading-relaxed text-rose-300">
            SignBridge AI is an assistive communication aid for immediate in-person display and speech synthesis.
            It is <strong>NOT a replacement for emergency telephone services</strong>. If you are in immediate danger or life-threatening distress, dial <strong>112</strong> (National Emergency Helpline) or ask a bystander to call 112 immediately.
          </p>
        </div>
      </div>

      {/* Primary Message Card (Huge & High-Contrast) */}
      <div className="bg-slate-900 border-2 border-rose-500/50 rounded-3xl p-6 sm:p-8 space-y-5 shadow-2xl">
        <div className="flex items-center justify-between">
          <span className="text-xs font-bold text-rose-400 uppercase tracking-widest flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-rose-500 animate-ping"></span> Active Emergency Message
          </span>
          <span className="text-xs text-slate-400">Click phrase below or edit freely</span>
        </div>

        <textarea
          rows={3}
          value={activeMessage}
          onChange={(e) => setActiveMessage(e.target.value)}
          className="w-full p-4 rounded-2xl bg-black border-2 border-slate-700 text-white font-extrabold text-xl sm:text-2xl leading-relaxed focus:outline-none focus:border-rose-500 selection:bg-rose-500"
          aria-label="Active emergency message text"
        />

        {/* Big Action Buttons */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <button
            onClick={handleSpeakAloud}
            className="py-4 px-6 rounded-2xl bg-rose-600 hover:bg-rose-500 text-white font-extrabold text-base sm:text-lg flex items-center justify-center space-x-3 shadow-xl shadow-rose-600/30 transition-transform hover:scale-101"
          >
            <Volume2 className="w-6 h-6" />
            <span>PLAY LOUD VOICE ALARM</span>
          </button>

          <button
            onClick={handleCopy}
            className="py-4 px-6 rounded-2xl bg-slate-800 hover:bg-slate-700 text-white font-bold text-sm sm:text-base flex items-center justify-center space-x-2 border border-slate-700"
          >
            {copied ? <Check className="w-5 h-5 text-emerald-400" /> : <Copy className="w-5 h-5" />}
            <span>{copied ? 'Copied to Clipboard' : 'Copy Message for SMS / WhatsApp'}</span>
          </button>
        </div>
      </div>

      {/* Preset Emergency Quick Tiles */}
      <div className="space-y-4">
        <h2 className="text-base font-bold text-white uppercase tracking-wider">
          One-Touch Preset Emergency Statements
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {emergencyPresets.map((preset) => {
            const Icon = preset.icon;
            return (
              <button
                key={preset.title}
                onClick={() => {
                  setActiveMessage(preset.text);
                  speakText(preset.text, 'en');
                }}
                className="p-5 rounded-2xl bg-slate-900 border border-slate-800 hover:border-rose-500/60 hover:bg-slate-850 text-left space-y-2 transition-all group"
              >
                <div className="flex items-center justify-between">
                  <div className="p-2.5 rounded-xl bg-slate-800 text-rose-400 group-hover:scale-105 transition-transform">
                    <Icon className="w-5 h-5" />
                  </div>
                  <span className="text-[11px] font-mono font-semibold px-2 py-0.5 rounded bg-slate-950 text-slate-400">
                    Touch to Speak
                  </span>
                </div>
                <h3 className="text-base font-bold text-white group-hover:text-rose-300">{preset.title}</h3>
                <p className="text-xs text-slate-400 leading-relaxed line-clamp-2">{preset.text}</p>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
};
