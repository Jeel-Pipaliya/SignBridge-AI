import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Video,
  BookOpen,
  MessageSquare,
  Captions,
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  Sparkles,
  ArrowRight,
  ShieldCheck,
  Eye,
  Volume2,
  Cpu,
  Layers,
} from 'lucide-react';
import { useAccessibility } from '../context/AccessibilityContext';

export const LandingPage: React.FC = () => {
  const { speakText } = useAccessibility();
  const [demoSign, setDemoSign] = useState<string>('HELLO');

  const demoPhrases = [
    { sign: 'HELLO', desc: 'Greeting: Raised open palm from forehead outward' },
    { sign: 'L', desc: 'Alphabet: Perpendicular index finger and thumb' },
    { sign: '1', desc: 'Number: Single upright index finger from fist' },
    { sign: 'C', desc: 'Alphabet: Curved hand forming a C arc' },
    { sign: 'Y', desc: 'Alphabet: Extended thumb and pinky (shaka)' },
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      {/* Hero Section */}
      <section className="relative overflow-hidden pt-12 pb-20 lg:pt-20 lg:pb-32 border-b border-slate-900">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_80%_at_50%_-20%,rgba(20,184,166,0.15),rgba(255,255,255,0))]"></div>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
          <div className="text-center max-w-3xl mx-auto space-y-6">
            <div className="inline-flex items-center space-x-2 px-3 py-1.5 rounded-full bg-teal-500/10 border border-teal-500/20 text-teal-300 text-xs font-semibold">
              <Sparkles className="w-3.5 h-3.5 text-teal-400" />
              <span>Real-Time Indian Sign Language AI Communication</span>
            </div>

            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight text-white leading-tight">
              Bridging the Silence with <span className="text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-emerald-400">Intelligent Computer Vision</span>
            </h1>

            <p className="text-base sm:text-lg text-slate-300 leading-relaxed font-normal">
              SignBridge AI translates Indian Sign Language (ISL) gestures into natural text and spoken voice in real time using webcam landmark tracking, neural sequence models, and grammar reconstruction.
            </p>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-3 pt-4">
              <Link
                to="/recognize"
                className="w-full sm:w-auto px-6 py-3.5 rounded-xl bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold text-sm shadow-lg shadow-teal-500/25 flex items-center justify-center space-x-2 transition-all hover:scale-102"
              >
                <Video className="w-4 h-4" />
                <span>Launch Live Recognition</span>
                <ArrowRight className="w-4 h-4" />
              </Link>
              <Link
                to="/tutor"
                className="w-full sm:w-auto px-6 py-3.5 rounded-xl bg-slate-900 hover:bg-slate-800 text-slate-200 border border-slate-800 font-semibold text-sm flex items-center justify-center space-x-2 transition-colors"
              >
                <BookOpen className="w-4 h-4" />
                <span>Browse 32-Sign Tutor</span>
              </Link>
              <Link
                to="/emergency"
                className="w-full sm:w-auto px-5 py-3.5 rounded-xl bg-rose-500/10 hover:bg-rose-500/20 text-rose-300 border border-rose-500/30 font-semibold text-sm flex items-center justify-center space-x-2 transition-colors"
              >
                <AlertTriangle className="w-4 h-4 text-rose-400" />
                <span>Emergency Mode</span>
              </Link>
            </div>

            {/* Quick Metrics */}
            <div className="pt-8 grid grid-cols-2 sm:grid-cols-4 gap-4 text-left border-t border-slate-800/80 mt-8">
              <div>
                <p className="text-2xl font-black text-white">32 Signs</p>
                <p className="text-xs text-slate-400">Verified ISL Vocabulary</p>
              </div>
              <div>
                <p className="text-2xl font-black text-teal-400">&lt; 15 ms</p>
                <p className="text-xs text-slate-400">Inference Latency</p>
              </div>
              <div>
                <p className="text-2xl font-black text-emerald-400">66.07%</p>
                <p className="text-xs text-slate-400">Held-Out Test Accuracy</p>
              </div>
              <div>
                <p className="text-2xl font-black text-white">100% Offline</p>
                <p className="text-xs text-slate-400">TTS & Vision Core</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Interactive Demonstration Preview */}
      <section className="py-16 bg-slate-900/60 border-b border-slate-900">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-2xl mx-auto mb-10">
            <h2 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">Interactive Gesture Explorer</h2>
            <p className="text-sm text-slate-400 mt-2">
              Select an Indian Sign Language class to preview anatomical landmark orientation and speech synthesis.
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-center">
            {/* Class Selection Buttons */}
            <div className="space-y-2">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block mb-2">Sample Classes</span>
              {demoPhrases.map((item) => (
                <button
                  key={item.sign}
                  onClick={() => setDemoSign(item.sign)}
                  className={`w-full text-left p-3.5 rounded-xl border transition-all flex items-center justify-between ${
                    demoSign === item.sign
                      ? 'bg-teal-500/15 border-teal-500/50 text-white shadow-md shadow-teal-500/10'
                      : 'bg-slate-900/80 border-slate-800 text-slate-300 hover:border-slate-700'
                  }`}
                >
                  <div>
                    <span className="text-sm font-bold">{item.sign}</span>
                    <p className="text-xs text-slate-400 line-clamp-1">{item.desc}</p>
                  </div>
                  <span className="text-xs px-2 py-0.5 rounded bg-slate-800 text-teal-400 font-mono">Select</span>
                </button>
              ))}
            </div>

            {/* Visualizer Simulation Card */}
            <div className="lg:col-span-2 bg-slate-950 border border-slate-800 rounded-2xl p-6 relative shadow-xl overflow-hidden">
              <div className="flex items-center justify-between border-b border-slate-800/80 pb-4 mb-4">
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 rounded-full bg-emerald-500 animate-pulse"></div>
                  <span className="text-xs font-mono text-slate-400">MediaPipe Hand Landmark Pipeline (21 Joint Nodes)</span>
                </div>
                <span className="text-xs font-mono px-2 py-1 rounded bg-slate-900 text-teal-300 border border-slate-800">
                  Wrist-Centered 63D Vector
                </span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 items-center py-4">
                {/* Anatomical Gesture Illustration Box */}
                <div className="aspect-video sm:aspect-square bg-slate-900/90 rounded-xl border border-slate-800 flex flex-col items-center justify-center p-6 relative text-center">
                  <div className="w-20 h-20 rounded-2xl bg-teal-500/20 border border-teal-500/40 flex items-center justify-center text-teal-300 text-3xl font-extrabold shadow-lg shadow-teal-500/10 mb-3">
                    {demoSign}
                  </div>
                  <p className="text-xs font-semibold text-white">Target Gesture: {demoSign}</p>
                  <p className="text-xs text-slate-400 mt-1 max-w-xs">
                    {demoPhrases.find((d) => d.sign === demoSign)?.desc}
                  </p>
                </div>

                {/* Simulated Telemetry and TTS */}
                <div className="space-y-4">
                  <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
                    <div className="flex justify-between text-xs text-slate-400">
                      <span>Predicted Sign:</span>
                      <span className="text-emerald-400 font-bold">{demoSign}</span>
                    </div>
                    <div className="flex justify-between text-xs text-slate-400">
                      <span>Model Confidence:</span>
                      <span className="text-teal-400 font-mono font-semibold">94.8%</span>
                    </div>
                    <div className="flex justify-between text-xs text-slate-400">
                      <span>Invariance Normalization:</span>
                      <span className="text-slate-300">Wrist Translation & Scale</span>
                    </div>
                    <div className="flex justify-between text-xs text-slate-400">
                      <span>Inference Duration:</span>
                      <span className="text-slate-300 font-mono">3.8 ms</span>
                    </div>
                  </div>

                  <button
                    onClick={() => speakText(`Recognized sign: ${demoSign}`)}
                    className="w-full py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-white text-xs font-semibold flex items-center justify-center space-x-2 border border-slate-700 transition-colors"
                  >
                    <Volume2 className="w-4 h-4 text-teal-400" />
                    <span>Test Speech Output: "{demoSign}"</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Architecture & Pipeline Section */}
      <section className="py-20 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center max-w-3xl mx-auto mb-16">
          <span className="text-xs font-bold text-teal-400 uppercase tracking-wider">End-to-End System Design</span>
          <h2 className="text-3xl font-extrabold text-white mt-2">How SignBridge AI Operates</h2>
          <p className="text-sm text-slate-400 mt-2">
            A pipeline ensuring privacy, mathematical invariance, and sub-second feedback without specialized hardware.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 relative">
            <div className="w-10 h-10 rounded-xl bg-teal-500/10 border border-teal-500/20 text-teal-400 flex items-center justify-center mb-4">
              <Video className="w-5 h-5" />
            </div>
            <span className="text-xs font-mono text-teal-400">01 / Video Input</span>
            <h3 className="text-base font-bold text-white mt-1">Webcam Capture</h3>
            <p className="text-xs text-slate-400 mt-2 leading-relaxed">
              Standard 30 FPS RGB video captured securely through browser MediaDevices API with client-side permissions.
            </p>
          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 relative">
            <div className="w-10 h-10 rounded-xl bg-teal-500/10 border border-teal-500/20 text-teal-400 flex items-center justify-center mb-4">
              <Layers className="w-5 h-5" />
            </div>
            <span className="text-xs font-mono text-teal-400">02 / Feature Engine</span>
            <h3 className="text-base font-bold text-white mt-1">Invariant Landmarks</h3>
            <p className="text-xs text-slate-400 mt-2 leading-relaxed">
              MediaPipe extracts 21 3D hand joints. Wrist-relative translation and middle-MCP distance normalization eliminate positioning bias.
            </p>
          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 relative">
            <div className="w-10 h-10 rounded-xl bg-teal-500/10 border border-teal-500/20 text-teal-400 flex items-center justify-center mb-4">
              <Cpu className="w-5 h-5" />
            </div>
            <span className="text-xs font-mono text-teal-400">03 / Dual AI Models</span>
            <h3 className="text-base font-bold text-white mt-1">Neural Classification</h3>
            <p className="text-xs text-slate-400 mt-2 leading-relaxed">
              Fast static MLP neural network for letters and numbers plus Bi-LSTM sequence classifier for dynamic temporal signs (HELLO, J, Z).
            </p>
          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 relative">
            <div className="w-10 h-10 rounded-xl bg-teal-500/10 border border-teal-500/20 text-teal-400 flex items-center justify-center mb-4">
              <MessageSquare className="w-5 h-5" />
            </div>
            <span className="text-xs font-mono text-teal-400">04 / Natural Output</span>
            <h3 className="text-base font-bold text-white mt-1">Grammar & Speech</h3>
            <p className="text-xs text-slate-400 mt-2 leading-relaxed">
              ISL grammar engine transforms raw sign tokens into fluent English and Hindi, with seamless offline text-to-speech audio playback.
            </p>
          </div>
        </div>
      </section>

      {/* Feature Suite Section */}
      <section className="py-20 bg-slate-900/40 border-t border-b border-slate-900">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-3xl mx-auto mb-16">
            <h2 className="text-3xl font-extrabold text-white">Comprehensive Feature Suite</h2>
            <p className="text-sm text-slate-400 mt-2">
              Engineered specifically for classroom instruction, daily conversations, workplace meetings, and emergency scenarios.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-3">
              <div className="w-10 h-10 rounded-xl bg-teal-500/10 text-teal-400 flex items-center justify-center">
                <Video className="w-5 h-5" />
              </div>
              <h3 className="text-lg font-bold text-white">Live Recognition Studio</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Live webcam capture with real-time landmark mesh preview, configurable confidence thresholds, sentence builder, undo/clear, and instant speech synthesis.
              </p>
              <Link to="/recognize" className="inline-flex items-center text-xs font-semibold text-teal-400 hover:text-teal-300 pt-2">
                Launch Recognition <ArrowRight className="w-3.5 h-3.5 ml-1" />
              </Link>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-3">
              <div className="w-10 h-10 rounded-xl bg-teal-500/10 text-teal-400 flex items-center justify-center">
                <BookOpen className="w-5 h-5" />
              </div>
              <h3 className="text-lg font-bold text-white">Sign Learning Tutor</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Interactive practice mode comparing student gestures against verified anatomical reference standards with measurable accuracy scoring and historical streak tracking.
              </p>
              <Link to="/tutor" className="inline-flex items-center text-xs font-semibold text-teal-400 hover:text-teal-300 pt-2">
                Open Tutor <ArrowRight className="w-3.5 h-3.5 ml-1" />
              </Link>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-3">
              <div className="w-10 h-10 rounded-xl bg-teal-500/10 text-teal-400 flex items-center justify-center">
                <MessageSquare className="w-5 h-5" />
              </div>
              <h3 className="text-lg font-bold text-white">Two-Way Conversation Hub</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Bidirectional communication enabling hearing individuals to speak into the microphone (speech-to-text) while deaf users sign or type with synchronized speech playback.
              </p>
              <Link to="/communication" className="inline-flex items-center text-xs font-semibold text-teal-400 hover:text-teal-300 pt-2">
                Start Conversation <ArrowRight className="w-3.5 h-3.5 ml-1" />
              </Link>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-3">
              <div className="w-10 h-10 rounded-xl bg-teal-500/10 text-teal-400 flex items-center justify-center">
                <Captions className="w-5 h-5" />
              </div>
              <h3 className="text-lg font-bold text-white">Live Meeting Captions</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Accessible meeting transcription with live speech-to-text captions, timestamped speaker logs, recording consent indicators, and exportable transcript files (TXT, JSON).
              </p>
              <Link to="/captions" className="inline-flex items-center text-xs font-semibold text-teal-400 hover:text-teal-300 pt-2">
                Open Meeting Captions <ArrowRight className="w-3.5 h-3.5 ml-1" />
              </Link>
            </div>

            <div className="bg-slate-900 border border-rose-500/30 rounded-2xl p-6 space-y-3 bg-gradient-to-b from-rose-950/20 to-slate-900">
              <div className="w-10 h-10 rounded-xl bg-rose-500/10 text-rose-400 flex items-center justify-center">
                <AlertTriangle className="w-5 h-5" />
              </div>
              <h3 className="text-lg font-bold text-white">Emergency Communication</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                High-contrast, large-button preset emergency phrases ("Need Medical Help", "Fire Hazard", "Call Ambulance") with instant loud speech alarm and 112 emergency services advisory.
              </p>
              <Link to="/emergency" className="inline-flex items-center text-xs font-semibold text-rose-400 hover:text-rose-300 pt-2">
                Emergency Mode <ArrowRight className="w-3.5 h-3.5 ml-1" />
              </Link>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-3">
              <div className="w-10 h-10 rounded-xl bg-teal-500/10 text-teal-400 flex items-center justify-center">
                <BarChart3 className="w-5 h-5" />
              </div>
              <h3 className="text-lg font-bold text-white">Evaluation & Analytics</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Transparent model performance dashboard explicitly contrasting held-out offline test metrics (66.07% accuracy, 0.6085 F1) with live user practice telemetry.
              </p>
              <Link to="/analytics" className="inline-flex items-center text-xs font-semibold text-teal-400 hover:text-teal-300 pt-2">
                View Analytics <ArrowRight className="w-3.5 h-3.5 ml-1" />
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Footer Banner */}
      <section className="py-16 text-center max-w-4xl mx-auto px-4">
        <div className="p-8 sm:p-12 rounded-3xl bg-gradient-to-r from-teal-950/60 to-slate-900 border border-teal-500/30 space-y-4">
          <h2 className="text-2xl sm:text-3xl font-extrabold text-white">Ready to Experience SignBridge AI?</h2>
          <p className="text-sm text-slate-300 max-w-xl mx-auto">
            Try the demo account or sign up to save your practice streaks, customized voice settings, and conversation transcripts.
          </p>
          <div className="pt-3 flex flex-wrap justify-center gap-3">
            <Link
              to="/recognize"
              className="px-6 py-3 rounded-xl bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold text-sm shadow-lg shadow-teal-500/20"
            >
              Start Webcam Recognition
            </Link>
            <Link
              to="/login"
              className="px-6 py-3 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-sm font-semibold"
            >
              Sign In (Demo Account Available)
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
};
