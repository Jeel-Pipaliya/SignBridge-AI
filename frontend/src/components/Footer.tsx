import React from 'react';
import { Link } from 'react-router-dom';
import { Heart, Shield, Cpu, Accessibility } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer className="bg-slate-950 border-t border-slate-900 text-slate-400 text-sm py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-8">
          {/* Brand Info */}
          <div className="space-y-3 md:col-span-1">
            <div className="flex items-center space-x-2">
              <div className="w-7 h-7 rounded-lg bg-teal-500 flex items-center justify-center">
                <span className="text-slate-950 font-bold text-sm">S</span>
              </div>
              <span className="text-white font-bold text-base">SignBridge AI</span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Empowering the Deaf and hard-of-hearing community with real-time Indian Sign Language (ISL) vision recognition, educational tutoring, and bidirectional voice communication.
            </p>
            <div className="flex items-center space-x-2 text-xs text-teal-400 pt-1">
              <Accessibility className="w-4 h-4" />
              <span>Accessibility-First Architecture (WCAG 2.1 AA)</span>
            </div>
          </div>

          {/* Quick Tools */}
          <div>
            <h4 className="text-white font-semibold text-xs tracking-wider uppercase mb-3">Core Modules</h4>
            <ul className="space-y-2 text-xs">
              <li><Link to="/recognize" className="hover:text-teal-400 transition-colors">Live ISL Recognition</Link></li>
              <li><Link to="/tutor" className="hover:text-teal-400 transition-colors">32-Sign Learning Tutor</Link></li>
              <li><Link to="/communication" className="hover:text-teal-400 transition-colors">Voice & Speech Hub</Link></li>
              <li><Link to="/captions" className="hover:text-teal-400 transition-colors">Live Meeting Captions</Link></li>
              <li><Link to="/emergency" className="hover:text-teal-400 text-rose-400 transition-colors">Emergency Assist</Link></li>
            </ul>
          </div>

          {/* Machine Learning & Models */}
          <div>
            <h4 className="text-white font-semibold text-xs tracking-wider uppercase mb-3">AI Architecture</h4>
            <ul className="space-y-2 text-xs">
              <li className="flex items-center gap-1.5"><Cpu className="w-3.5 h-3.5 text-teal-400" /> 32-Class ISL MLP (66.07% Test Acc)</li>
              <li className="flex items-center gap-1.5"><Cpu className="w-3.5 h-3.5 text-teal-400" /> Bi-LSTM Temporal Sequence Model</li>
              <li><span>Invariant Translation & Scale Normalization</span></li>
              <li><span>SOV to SVO Grammar Reconstruction</span></li>
              <li><span>Hindi Transliteration & TTS</span></li>
            </ul>
          </div>

          {/* Disclaimers & Ethics */}
          <div>
            <h4 className="text-white font-semibold text-xs tracking-wider uppercase mb-3">Ethics & Privacy</h4>
            <p className="text-xs text-slate-500 leading-relaxed mb-3">
              Webcam video and audio streams are processed locally in-browser or streamed ephemerally for landmark calculation. No biometric footage is stored without explicit user consent.
            </p>
            <div className="flex items-center gap-1 text-xs text-slate-500">
              <Shield className="w-3.5 h-3.5 text-emerald-400" />
              <span>GDPR Privacy Erasure Supported</span>
            </div>
          </div>
        </div>

        <div className="pt-8 border-t border-slate-900 flex flex-col sm:flex-row items-center justify-between text-xs text-slate-500">
          <p>© {new Date().getFullYear()} SignBridge AI. Open-source under MIT License.</p>
          <p className="mt-2 sm:mt-0 flex items-center gap-1">
            Built with <Heart className="w-3 h-3 text-rose-500 inline fill-rose-500" /> for the Indian Deaf Community
          </p>
        </div>
      </div>
    </footer>
  );
};
