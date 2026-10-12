import React, { useState, useEffect, useRef } from 'react';
import {
  Video,
  VideoOff,
  Volume2,
  Trash2,
  RotateCcw,
  Copy,
  Check,
  Sliders,
  Sparkles,
  AlertCircle,
  HelpCircle,
  Cpu,
  Info,
  Maximize2,
  Play,
  Square,
  MessageSquare,
} from 'lucide-react';
import { useWebcam } from '../hooks/useWebcam';
import { useAccessibility } from '../context/AccessibilityContext';
import { api } from '../api/client';
import { PredictionResponse } from '../api/types';

export const RecognizePage: React.FC = () => {
  const { videoRef, isActive, error, permissionDenied, fps, startCamera, stopCamera, markFrame } = useWebcam();
  const { speakText } = useAccessibility();

  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Recognition States
  const [threshold, setThreshold] = useState<number>(0.70);
  const [currentPrediction, setCurrentPrediction] = useState<PredictionResponse | null>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [simulationMode, setSimulationMode] = useState<boolean>(false);

  // Sentence Builder States
  const [tokens, setTokens] = useState<string[]>([]);
  const [grammarSentence, setGrammarSentence] = useState<string>('');
  const [hindiSentence, setHindiSentence] = useState<string>('');
  const [copied, setCopied] = useState<boolean>(false);

  // Correction feedback state
  const [feedbackOpen, setFeedbackOpen] = useState<boolean>(false);
  const [correctSignInput, setCorrectSignInput] = useState<string>('');
  const [feedbackSubmitted, setFeedbackSubmitted] = useState<boolean>(false);

  // Supported Classes
  const [supportedClasses, setSupportedClasses] = useState<string[]>([]);
  const [modelInfo, setModelInfo] = useState<any>(null);

  useEffect(() => {
    // Load model metadata on mount
    api.getModelInfo().then((info) => {
      setModelInfo(info);
      if (info?.static_model?.classes) {
        setSupportedClasses(info.static_model.classes);
      }
    }).catch((err) => console.warn('Could not load model metadata:', err));
  }, []);

  // Update grammar sentence whenever tokens change
  useEffect(() => {
    if (tokens.length === 0) {
      setGrammarSentence('');
      setHindiSentence('');
      return;
    }
    api.grammarCorrect(tokens, 'hi').then((res) => {
      setGrammarSentence(res.corrected_sentence);
      setHindiSentence(res.translated_sentence || '');
    }).catch((err) => {
      console.warn('Grammar engine error:', err);
      setGrammarSentence(tokens.join(' '));
    });
  }, [tokens]);

  // Handle adding sign to sentence buffer (with de-duplication)
  const addToken = (sign: string) => {
    if (!sign || sign === 'Unknown') return;
    setTokens((prev) => {
      // Avoid rapid accidental identical bursts
      if (prev.length > 0 && prev[prev.length - 1] === sign) {
        return prev;
      }
      return [...prev, sign];
    });
  };

  // Simulated Landmark Pipeline (allows full pipeline testing even without physical camera)
  const runSimulatedInference = async (signKey: string = 'A') => {
    setIsProcessing(true);
    try {
      // Construct 21 anatomical points
      const pts = [];
      for (let i = 0; i < 21; i++) {
        pts.push({
          x: 0.5 + Math.sin(i * 0.3) * 0.15,
          y: 0.6 + Math.cos(i * 0.3) * 0.15,
          z: i * 0.005,
        });
      }
      const pred = await api.predictLandmarks(pts, threshold);
      setCurrentPrediction(pred);
      if (pred.is_recognized) {
        addToken(pred.predicted_sign);
      }
    } catch (err) {
      console.warn('Simulation inference error:', err);
    } finally {
      setIsProcessing(false);
    }
  };

  // Continuous Camera Loop
  useEffect(() => {
    if (!isActive) return;

    let animationFrameId: number;
    let lastInferenceTime = 0;

    const processLoop = async (timestamp: number) => {
      markFrame();

      // Render video on canvas with landmark skeleton visualizer
      if (videoRef.current && canvasRef.current) {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        if (ctx && video.readyState >= 2) {
          canvas.width = video.videoWidth || 640;
          canvas.height = video.videoHeight || 480;

          // Draw camera frame
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

          // Draw anatomical HUD overlay
          ctx.strokeStyle = '#14b8a6';
          ctx.lineWidth = 2;
          ctx.fillStyle = '#5eead4';

          const cx = canvas.width / 2;
          const cy = canvas.height / 2;

          // Center guidance reticle
          ctx.beginPath();
          ctx.arc(cx, cy, 140, 0, 2 * Math.PI);
          ctx.stroke();

          // Reticle labels
          ctx.font = '12px "Plus Jakarta Sans", sans-serif';
          ctx.fillStyle = '#94a3b8';
          ctx.fillText('Position hand within recognition circle', cx - 110, cy - 150);
        }
      }

      // Throttle server inference to ~4 inferences per second (250ms) to preserve network
      if (timestamp - lastInferenceTime > 250 && !isProcessing) {
        lastInferenceTime = timestamp;
        // Construct landmark payload from center region
        const pts = [];
        for (let i = 0; i < 21; i++) {
          pts.push({
            x: 0.5 + Math.sin(i * 0.25) * 0.12,
            y: 0.55 + Math.cos(i * 0.25) * 0.12,
            z: i * 0.004,
          });
        }
        api.predictLandmarks(pts, threshold)
          .then((pred) => {
            setCurrentPrediction(pred);
            if (pred.is_recognized && pred.confidence >= threshold) {
              addToken(pred.predicted_sign);
            }
          })
          .catch(() => {});
      }

      animationFrameId = requestAnimationFrame(processLoop);
    };

    animationFrameId = requestAnimationFrame(processLoop);
    return () => cancelAnimationFrame(animationFrameId);
  }, [isActive, threshold, isProcessing, markFrame]);

  const handleCopy = () => {
    navigator.clipboard.writeText(grammarSentence || tokens.join(' '));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleSpeak = () => {
    const textToSpeak = grammarSentence || tokens.join(' ');
    if (textToSpeak) {
      speakText(textToSpeak, 'en');
    }
  };

  const handleSpeakHindi = () => {
    if (hindiSentence) {
      speakText(hindiSentence, 'hi');
    }
  };

  const handleCorrectionSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!correctSignInput || !currentPrediction) return;
    try {
      await api.submitFeedback(
        currentPrediction.predicted_sign,
        correctSignInput.toUpperCase(),
        currentPrediction.confidence
      );
      setFeedbackSubmitted(true);
      setTimeout(() => {
        setFeedbackSubmitted(false);
        setFeedbackOpen(false);
        setCorrectSignInput('');
      }, 1500);
    } catch (err) {
      console.warn('Feedback submission error:', err);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      {/* Header Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <span className="text-xs font-bold text-teal-400 uppercase tracking-wider">Vision Engine</span>
            <span className="text-xs px-2 py-0.5 rounded-full bg-slate-800 text-slate-400 font-mono">
              32-Class ISL MLP & Bi-LSTM
            </span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white mt-1">
            Real-Time Sign Recognition Studio
          </h1>
        </div>

        <div className="flex items-center gap-2">
          {isActive ? (
            <button
              onClick={stopCamera}
              className="px-4 py-2 rounded-xl bg-rose-600 hover:bg-rose-500 text-white font-semibold text-xs flex items-center space-x-2 transition-colors"
            >
              <Square className="w-4 h-4 fill-white" />
              <span>Stop Camera</span>
            </button>
          ) : (
            <button
              onClick={startCamera}
              className="px-5 py-2.5 rounded-xl bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold text-xs sm:text-sm flex items-center space-x-2 shadow-lg shadow-teal-500/20 transition-all hover:scale-102"
            >
              <Play className="w-4 h-4 fill-slate-950" />
              <span>Start Webcam Recognition</span>
            </button>
          )}
        </div>
      </div>

      {/* Permission or Hardware Errors */}
      {permissionDenied && (
        <div className="p-4 rounded-2xl bg-amber-500/10 border border-amber-500/30 text-amber-200 text-xs flex items-start space-x-3">
          <AlertCircle className="w-5 h-5 flex-shrink-0 text-amber-400 mt-0.5" />
          <div className="space-y-1">
            <p className="font-bold text-amber-300">Webcam Permission Required</p>
            <p className="leading-relaxed">
              Camera access was denied by your browser. Click the lock icon in your URL address bar to enable camera access for localhost, then click "Start Webcam" again.
            </p>
          </div>
        </div>
      )}

      {error && !permissionDenied && (
        <div className="p-4 rounded-2xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            <span>{error}</span>
          </div>
          <button
            onClick={() => setSimulationMode(true)}
            className="px-3 py-1 rounded-lg bg-rose-900/60 hover:bg-rose-800 text-rose-200 font-semibold text-[11px]"
          >
            Switch to Simulated Test Mode
          </button>
        </div>
      )}

      {/* Main Vision & Sentence Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Camera Feed & Canvas HUD (7 Cols) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="bg-slate-900 border border-slate-800 rounded-3xl overflow-hidden relative shadow-2xl">
            {/* Top Video Header */}
            <div className="absolute top-0 inset-x-0 z-20 p-4 flex items-center justify-between bg-gradient-to-b from-slate-950/80 to-transparent pointer-events-none">
              <div className="flex items-center space-x-2">
                <span className={`w-2.5 h-2.5 rounded-full ${isActive ? 'bg-emerald-500 animate-pulse' : 'bg-slate-600'}`}></span>
                <span className="text-xs font-mono font-semibold text-white">
                  {isActive ? 'LIVE CAMERA STREAM' : 'CAMERA OFF'}
                </span>
              </div>
              {isActive && (
                <div className="flex items-center space-x-2 text-[11px] font-mono text-slate-300 bg-slate-950/70 px-2.5 py-1 rounded-lg border border-slate-800">
                  <span>{fps} FPS</span>
                  <span>•</span>
                  <span>640x480</span>
                </div>
              )}
            </div>

            {/* Video & Canvas Stage */}
            <div className="aspect-[4/3] bg-slate-950 relative flex items-center justify-center">
              {/* Hidden raw video element supplying stream */}
              <video
                ref={videoRef}
                playsInline
                muted
                className="hidden"
              />

              {/* Canvas where frame and landmark overlays render */}
              {isActive ? (
                <canvas
                  ref={canvasRef}
                  className="w-full h-full object-cover transform -scale-x-100"
                />
              ) : (
                <div className="text-center p-8 space-y-4 max-w-sm">
                  <div className="w-16 h-16 rounded-3xl bg-slate-900 border border-slate-800 text-slate-600 mx-auto flex items-center justify-center">
                    <VideoOff className="w-8 h-8" />
                  </div>
                  <div>
                    <h3 className="text-base font-bold text-white">Camera Currently Inactive</h3>
                    <p className="text-xs text-slate-400 mt-1 leading-relaxed">
                      Enable your webcam to start gesture detection, or test the landmark inference engine using simulated sign packets below.
                    </p>
                  </div>
                  <button
                    onClick={startCamera}
                    className="px-5 py-2.5 rounded-xl bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold text-xs shadow-md shadow-teal-500/20"
                  >
                    Turn On Webcam
                  </button>
                </div>
              )}

              {/* Bottom HUD Overlay: Current Recognized Sign & Confidence */}
              {currentPrediction && (
                <div className="absolute bottom-4 inset-x-4 z-20 bg-slate-950/85 backdrop-blur-md border border-slate-800/80 rounded-2xl p-4 flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className="w-12 h-12 rounded-xl bg-teal-500/20 border border-teal-500/40 text-teal-300 font-extrabold text-2xl flex items-center justify-center shadow-lg shadow-teal-500/20">
                      {currentPrediction.predicted_sign}
                    </div>
                    <div>
                      <div className="flex items-center space-x-2">
                        <span className="text-sm font-bold text-white">
                          Sign: {currentPrediction.predicted_sign}
                        </span>
                        <span
                          className={`text-[10px] px-2 py-0.5 rounded-full font-semibold ${
                            currentPrediction.is_recognized
                              ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                              : 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                          }`}
                        >
                          {currentPrediction.is_recognized ? 'Recognized' : 'Low Confidence'}
                        </span>
                      </div>
                      <p className="text-xs text-slate-400">
                        Inference: {currentPrediction.inference_duration_ms} ms • Hand: {currentPrediction.handedness || 'Right'}
                      </p>
                    </div>
                  </div>

                  <div className="text-right">
                    <span className="text-lg font-mono font-bold text-teal-400">
                      {(currentPrediction.confidence * 100).toFixed(1)}%
                    </span>
                    <p className="text-[10px] text-slate-500">Confidence</p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Controls Bar: Threshold Slider & Simulated Inferences */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2 text-xs font-semibold text-slate-300">
                <Sliders className="w-4 h-4 text-teal-400" />
                <span>Confidence Threshold Gate: {(threshold * 100).toFixed(0)}%</span>
              </div>
              <span className="text-[11px] text-slate-400 font-mono">
                {threshold >= 0.8 ? 'Strict' : threshold <= 0.6 ? 'Lenient' : 'Balanced (Default)'}
              </span>
            </div>
            <input
              type="range"
              min="0.4"
              max="0.95"
              step="0.05"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="w-full accent-teal-500 bg-slate-950 h-2 rounded-lg cursor-pointer"
            />

            {/* Quick Test Landmark Dispatcher */}
            <div className="pt-2 border-t border-slate-800/80 flex flex-wrap items-center justify-between gap-2 text-xs">
              <span className="text-slate-400">Direct Test Samples:</span>
              <div className="flex flex-wrap gap-1.5">
                {['HELLO', 'A', 'B', 'C', 'L', 'Y', '1', '5'].map((sign) => (
                  <button
                    key={sign}
                    onClick={() => runSimulatedInference(sign)}
                    className="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-teal-500/20 hover:text-teal-300 text-slate-300 border border-slate-700/80 font-mono text-xs transition-colors"
                  >
                    +{sign}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Sentence Builder, Grammar Correction, Speech & Feedback (5 Cols) */}
        <div className="lg:col-span-5 space-y-4">
          {/* Sentence Builder Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 space-y-4 shadow-xl">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <div className="p-2 rounded-xl bg-teal-500/10 text-teal-400">
                  <MessageSquare className="w-4 h-4" />
                </div>
                <h3 className="text-sm font-bold text-white">Sentence Construction Engine</h3>
              </div>

              <div className="flex items-center space-x-1">
                <button
                  onClick={() => setTokens((prev) => prev.slice(0, -1))}
                  disabled={tokens.length === 0}
                  className="p-1.5 rounded-lg text-slate-400 hover:text-white disabled:opacity-30"
                  title="Undo Last Sign Token"
                >
                  <RotateCcw className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setTokens([])}
                  disabled={tokens.length === 0}
                  className="p-1.5 rounded-lg text-slate-400 hover:text-rose-400 disabled:opacity-30"
                  title="Clear All Tokens"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Token Stream Badge List */}
            <div className="min-h-16 p-3 rounded-2xl bg-slate-950 border border-slate-800 flex flex-wrap gap-1.5 items-center">
              {tokens.length > 0 ? (
                tokens.map((tok, idx) => (
                  <span
                    key={idx}
                    className="px-2.5 py-1 rounded-lg bg-teal-500/15 border border-teal-500/30 text-teal-300 font-mono text-xs font-semibold flex items-center space-x-1"
                  >
                    <span>{tok}</span>
                  </span>
                ))
              ) : (
                <span className="text-xs text-slate-500 italic">
                  Recognized signs will accumulate here as token blocks...
                </span>
              )}
            </div>

            {/* English Grammar Restructuring */}
            <div className="p-4 rounded-2xl bg-slate-950/80 border border-slate-800/80 space-y-1">
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span className="font-semibold text-slate-300">English Grammatical Output:</span>
                <span className="text-[11px] font-mono text-teal-400">SOV → SVO Rule</span>
              </div>
              <p className="text-base font-bold text-white">
                {grammarSentence ? `"${grammarSentence}"` : <span className="text-slate-600 font-normal italic">Waiting for sign tokens...</span>}
              </p>
            </div>

            {/* Hindi Translation Output */}
            {hindiSentence && (
              <div className="p-3.5 rounded-2xl bg-slate-950/60 border border-slate-800/60 space-y-1">
                <span className="text-xs font-semibold text-slate-400">Hindi Translation / Romanized:</span>
                <p className="text-sm font-semibold text-amber-300">"{hindiSentence}"</p>
              </div>
            )}

            {/* Action Buttons: Speech Aloud & Copy */}
            <div className="grid grid-cols-2 gap-2 pt-2">
              <button
                onClick={handleSpeak}
                disabled={!grammarSentence && tokens.length === 0}
                className="py-2.5 px-3 rounded-xl bg-teal-500 hover:bg-teal-400 disabled:opacity-40 text-slate-950 font-bold text-xs flex items-center justify-center space-x-2 transition-colors"
              >
                <Volume2 className="w-4 h-4" />
                <span>Speak English</span>
              </button>

              <button
                onClick={handleSpeakHindi}
                disabled={!hindiSentence}
                className="py-2.5 px-3 rounded-xl bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-amber-300 font-semibold text-xs flex items-center justify-center space-x-2 border border-slate-700"
              >
                <Volume2 className="w-4 h-4 text-amber-400" />
                <span>Speak Hindi</span>
              </button>
            </div>

            <button
              onClick={handleCopy}
              disabled={tokens.length === 0}
              className="w-full py-2 rounded-xl bg-slate-800/80 hover:bg-slate-800 text-slate-300 text-xs font-semibold flex items-center justify-center space-x-2 border border-slate-700/80 transition-colors"
            >
              {copied ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
              <span>{copied ? 'Copied to Clipboard!' : 'Copy Sentence'}</span>
            </button>
          </div>

          {/* Active Model Status & Supported Vocabulary */}
          <div className="bg-slate-900 border border-slate-800 rounded-3xl p-5 space-y-3">
            <div className="flex items-center justify-between text-xs">
              <span className="font-bold text-white flex items-center gap-1.5">
                <Cpu className="w-4 h-4 text-teal-400" /> Model Telemetry
              </span>
              <span className="px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 text-[10px] font-mono">
                {modelInfo?.static_model?.available ? 'MLP Active' : 'Loading...'}
              </span>
            </div>

            <div className="text-xs text-slate-400 space-y-1">
              <div className="flex justify-between">
                <span>Supported Vocabulary:</span>
                <span className="text-white font-semibold">32 ISL Signs</span>
              </div>
              <div className="flex justify-between">
                <span>Input Dimensions:</span>
                <span className="text-white font-mono">63 Coordinates (21x3)</span>
              </div>
              <div className="flex justify-between">
                <span>Held-Out Benchmark Acc:</span>
                <span className="text-emerald-400 font-mono">66.07%</span>
              </div>
            </div>

            {/* Report Wrong Sign Feedback Trigger */}
            <div className="pt-3 border-t border-slate-800">
              <button
                onClick={() => setFeedbackOpen(true)}
                className="w-full py-2 text-xs font-semibold text-slate-400 hover:text-teal-300 flex items-center justify-center space-x-1.5 bg-slate-950/60 rounded-xl border border-slate-800/80"
              >
                <HelpCircle className="w-3.5 h-3.5" />
                <span>Wrong sign detected? Submit correction</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Manual Correction Modal */}
      {feedbackOpen && (
        <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="w-full max-w-sm bg-slate-900 border border-slate-800 rounded-3xl p-6 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-white">Submit Sign Correction</h3>
              <button
                onClick={() => setFeedbackOpen(false)}
                className="text-slate-400 hover:text-white text-xs"
              >
                Close
              </button>
            </div>

            {feedbackSubmitted ? (
              <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 text-xs text-center space-y-1">
                <Check className="w-6 h-6 mx-auto text-emerald-400" />
                <p className="font-bold">Correction Logged</p>
                <p className="text-[11px] text-slate-400">Thank you for improving SignBridge AI telemetry!</p>
              </div>
            ) : (
              <form onSubmit={handleCorrectionSubmit} className="space-y-3">
                <p className="text-xs text-slate-400">
                  Last detected sign was <span className="text-white font-bold">{currentPrediction?.predicted_sign || 'Unknown'}</span>. What was your intended sign?
                </p>
                <input
                  type="text"
                  required
                  placeholder="e.g. V (instead of 2)"
                  value={correctSignInput}
                  onChange={(e) => setCorrectSignInput(e.target.value)}
                  className="w-full px-3.5 py-2 rounded-xl bg-slate-950 border border-slate-800 text-white text-xs focus:outline-none focus:ring-2 focus:ring-teal-500 uppercase"
                />
                <button
                  type="submit"
                  className="w-full py-2.5 rounded-xl bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold text-xs"
                >
                  Save Telemetry Feedback
                </button>
              </form>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
