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
    if (!sign || ['Unknown', 'No Hand Detected', 'Error'].includes(sign)) return;
    setTokens((prev) => {
      // Avoid rapid accidental identical bursts
      if (prev.length > 0 && prev[prev.length - 1] === sign) {
        return prev;
      }
      return [...prev, sign];
    });
  };

  // 21-joint anatomical skeleton connections for MediaPipe hand rendering
  const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],         // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8],         // Index finger
    [5, 9], [9, 10], [10, 11], [11, 12],    // Middle finger
    [9, 13], [13, 14], [14, 15], [15, 16],  // Ring finger
    [13, 17], [17, 18], [18, 19], [19, 20], // Pinky finger
    [0, 17],                                // Palm base
  ];

  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const stabilityCounterRef = useRef<{ sign: string; count: number }>({ sign: '', count: 0 });
  const latestPredictionRef = useRef<PredictionResponse | null>(null);

  // Direct Gesture Test Dispatcher (allows immediate sign testing without requiring camera)
  const runSimulatedInference = async (signKey: string = 'A') => {
    setIsProcessing(true);
    try {
      const mockPred: PredictionResponse = {
        predicted_sign: signKey,
        confidence: 0.94,
        is_recognized: true,
        is_supported_vocabulary: true,
        model_name: 'SignBridge_ISL_MLP_NeuralNet',
        model_version: '1.0',
        top_candidates: [
          { sign: signKey, confidence: 0.94 },
          { sign: 'B', confidence: 0.03 },
        ],
        inference_duration_ms: 15.2,
        handedness: 'Right',
        hand_detected: true,
      };
      setCurrentPrediction(mockPred);
      latestPredictionRef.current = mockPred;
      addToken(signKey);
    } finally {
      setIsProcessing(false);
    }
  };

  // Continuous Camera Loop with Real-Time Frame Capture & Landmark Skeleton
  useEffect(() => {
    if (!isActive) return;

    let animationFrameId: number;
    let lastInferenceTime = 0;
    let isRequestInFlight = false;

    // Offscreen canvas for downsampled video frame capture
    if (!captureCanvasRef.current) {
      captureCanvasRef.current = document.createElement('canvas');
      captureCanvasRef.current.width = 480;
      captureCanvasRef.current.height = 360;
    }

    const processLoop = async (timestamp: number) => {
      markFrame();

      const video = videoRef.current;
      const canvas = canvasRef.current;

      if (video && canvas && video.readyState >= 2) {
        const ctx = canvas.getContext('2d');
        if (ctx) {
          const vw = video.videoWidth || 640;
          const vh = video.videoHeight || 480;
          if (canvas.width !== vw || canvas.height !== vh) {
            canvas.width = vw;
            canvas.height = vh;
          }
          // Clear overlay canvas
          ctx.clearRect(0, 0, canvas.width, canvas.height);

          const pred = latestPredictionRef.current;

          // If hands and landmarks are detected by vision engine, draw anatomical skeleton
          if (pred?.hand_detected && pred.landmarks && pred.landmarks.length === 21) {
            const lms = pred.landmarks;

            // Draw glowing cyan bone lines
            ctx.save();
            ctx.strokeStyle = '#06b6d4';
            ctx.lineWidth = 3;
            ctx.shadowColor = '#22d3ee';
            ctx.shadowBlur = 10;
            ctx.lineCap = 'round';

            for (const [p1, p2] of HAND_CONNECTIONS) {
              if (lms[p1] && lms[p2]) {
                ctx.beginPath();
                ctx.moveTo(lms[p1].x * canvas.width, lms[p1].y * canvas.height);
                ctx.lineTo(lms[p2].x * canvas.width, lms[p2].y * canvas.height);
                ctx.stroke();
              }
            }
            ctx.restore();

            // Draw joint nodes (amber for fingertips, teal/cyan for joints)
            for (let i = 0; i < lms.length; i++) {
              const isTip = [4, 8, 12, 16, 20].includes(i);
              ctx.beginPath();
              ctx.arc(lms[i].x * canvas.width, lms[i].y * canvas.height, isTip ? 6 : 4, 0, 2 * Math.PI);
              ctx.fillStyle = isTip ? '#f59e0b' : '#38bdf8';
              ctx.fill();
              ctx.strokeStyle = '#ffffff';
              ctx.lineWidth = 1.5;
              ctx.stroke();
            }

            // Draw live tracking badge near wrist only for confirmed real-world sign
            const isRealBadge = pred.is_recognized &&
              pred.predicted_sign &&
              !['Unknown', 'No Hand Detected', 'Error', 'Unknown / Low Confidence'].includes(pred.predicted_sign) &&
              pred.confidence >= threshold;

            if (lms[0] && isRealBadge) {
              const wx = lms[0].x * canvas.width;
              const wy = Math.min(lms[0].y * canvas.height + 25, canvas.height - 20);
              ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
              ctx.roundRect ? ctx.roundRect(wx - 45, wy - 15, 90, 24, 6) : ctx.rect(wx - 45, wy - 15, 90, 24);
              ctx.fill();
              ctx.fillStyle = '#4ade80';
              ctx.font = 'bold 12px "Plus Jakarta Sans", sans-serif';
              ctx.textAlign = 'center';
              ctx.fillText(`${pred.predicted_sign} (${(pred.confidence * 100).toFixed(0)}%)`, wx, wy + 2);
            }
          } else {
            // Draw anatomical center guidance reticle when hand is not yet detected
            const cx = canvas.width / 2;
            const cy = canvas.height / 2;

            ctx.save();
            ctx.strokeStyle = 'rgba(20, 184, 166, 0.4)';
            ctx.lineWidth = 2;
            ctx.setLineDash([8, 8]);
            ctx.beginPath();
            ctx.arc(cx, cy, 130, 0, 2 * Math.PI);
            ctx.stroke();
            ctx.restore();

            ctx.font = '12px "Plus Jakarta Sans", sans-serif';
            ctx.fillStyle = '#cbd5e1';
            ctx.textAlign = 'center';
            ctx.fillText('Position hand in camera view to recognize ISL', cx, cy - 145);
          }
        }
      }

      // Send real video frame to backend MediaPipe inference engine every ~180ms
      if (timestamp - lastInferenceTime > 180 && !isRequestInFlight) {
        lastInferenceTime = timestamp;

        if (video && captureCanvasRef.current && video.readyState >= 2) {
          isRequestInFlight = true;
          const capCanvas = captureCanvasRef.current;
          const capCtx = capCanvas.getContext('2d');

          if (capCtx) {
            capCtx.drawImage(video, 0, 0, capCanvas.width, capCanvas.height);
            const frameJpeg = capCanvas.toDataURL('image/jpeg', 0.5);

            api.predictFrame(frameJpeg, threshold)
              .then((pred) => {
                setCurrentPrediction(pred);
                latestPredictionRef.current = pred;

                // Token stability gating: require 2 stable consecutive detections of the real-world gesture
                if (pred.is_recognized && pred.predicted_sign && !['Unknown', 'No Hand Detected', 'Error'].includes(pred.predicted_sign)) {
                  if (stabilityCounterRef.current.sign === pred.predicted_sign) {
                    stabilityCounterRef.current.count += 1;
                    if (stabilityCounterRef.current.count >= 2) {
                      addToken(pred.predicted_sign);
                      stabilityCounterRef.current = { sign: '', count: 0 };
                    }
                  } else {
                    stabilityCounterRef.current = { sign: pred.predicted_sign, count: 1 };
                  }
                } else {
                  stabilityCounterRef.current = { sign: '', count: 0 };
                }
              })
              .catch((err) => {
                console.warn('Real-time frame prediction error:', err);
              })
              .finally(() => {
                isRequestInFlight = false;
              });
          } else {
            isRequestInFlight = false;
          }
        }
      }

      animationFrameId = requestAnimationFrame(processLoop);
    };

    animationFrameId = requestAnimationFrame(processLoop);
    return () => {
      cancelAnimationFrame(animationFrameId);
      latestPredictionRef.current = null;
    };
  }, [isActive, threshold, markFrame]);


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
            <div className="aspect-[4/3] bg-slate-950 relative flex items-center justify-center overflow-hidden">
              {/* Hardware accelerated video element */}
              <video
                ref={videoRef}
                playsInline
                muted
                autoPlay
                className={`w-full h-full object-cover transform -scale-x-100 ${
                  isActive ? 'block' : 'hidden'
                }`}
              />

              {/* Canvas where landmark overlays render on top */}
              <canvas
                ref={canvasRef}
                className={`absolute inset-0 w-full h-full pointer-events-none transform -scale-x-100 ${
                  isActive ? 'block' : 'hidden'
                }`}
              />

              {!isActive && (
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

              {/* Bottom HUD Overlay: Only show when a real-world sign is recognized with high confidence */}
              {currentPrediction &&
              currentPrediction.is_recognized &&
              currentPrediction.confidence >= threshold &&
              !['Unknown', 'No Hand Detected', 'Error', 'Unknown / Low Confidence'].includes(currentPrediction.predicted_sign) ? (
                <div className="absolute bottom-4 inset-x-4 z-20 bg-slate-950/90 backdrop-blur-md border border-teal-500/40 rounded-2xl p-4 flex items-center justify-between shadow-xl">
                  <div className="flex items-center space-x-3">
                    <div className="min-w-12 h-12 px-3 rounded-xl bg-teal-500/20 border border-teal-500/40 text-teal-300 font-extrabold text-2xl flex items-center justify-center shadow-lg shadow-teal-500/20">
                      {currentPrediction.predicted_sign}
                    </div>
                    <div>
                      <div className="flex items-center space-x-2">
                        <span className="text-sm font-bold text-white">
                          Sign: {currentPrediction.predicted_sign}
                        </span>
                        <span className="text-[10px] px-2 py-0.5 rounded-full font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                          Recognized
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
              ) : isActive ? (
                <div className="absolute bottom-4 left-4 z-20 px-3.5 py-1.5 rounded-xl bg-slate-950/80 backdrop-blur-md border border-slate-800 text-xs text-slate-400 flex items-center space-x-2">
                  <span className="w-2 h-2 rounded-full bg-teal-400 animate-pulse"></span>
                  <span>Ready • Show real-world ISL sign</span>
                </div>
              ) : null}
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
                  {currentPrediction?.predicted_sign &&
                  !['Unknown', 'No Hand Detected', 'Error', 'Unknown / Low Confidence'].includes(currentPrediction.predicted_sign) ? (
                    <>Last detected sign was <span className="text-white font-bold">{currentPrediction.predicted_sign}</span>. What was your intended sign?</>
                  ) : (
                    <>What was your intended sign?</>
                  )}
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
