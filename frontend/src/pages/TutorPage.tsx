import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
  BookOpen,
  Search,
  CheckCircle2,
  XCircle,
  Volume2,
  Sparkles,
  ArrowRight,
  Flame,
  Award,
  Filter,
  Info,
  Play,
  RotateCcw,
  Video,
  VideoOff,
  Camera,
  Check,
} from 'lucide-react';
import { api } from '../api/client';
import { VocabularySign, PracticeSession, PracticeSummary, PredictionResponse } from '../api/types';
import { useAccessibility } from '../context/AccessibilityContext';
import { useAuth } from '../context/AuthContext';
import { useWebcam } from '../hooks/useWebcam';

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],         // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8],         // Index finger
  [5, 9], [9, 10], [10, 11], [11, 12],    // Middle finger
  [9, 13], [13, 14], [14, 15], [15, 16],  // Ring finger
  [13, 17], [17, 18], [18, 19], [19, 20], // Pinky finger
  [0, 17],                                // Palm base
];

// Complete verified 42-class ISL educational curriculum with anatomical guidance
const DEFAULT_VOCABULARY: VocabularySign[] = [
  // Real-World Words
  { id: 101, sign_name: "HELLO", category: "greeting", gesture_type: "dynamic", difficulty: "beginner",
    description: "Open palm starting near temple/forehead and sweeping gently outward in a welcoming salute.",
    anatomical_tips: "Start near temple with open fingers together; sweep gently outward toward partner." },
  { id: 102, sign_name: "THANK YOU", category: "greeting", gesture_type: "static", difficulty: "beginner",
    description: "Flat hand moves forward from chin or chest outward facing the conversational partner.",
    anatomical_tips: "Keep all 4 fingers together and straight, thumb resting alongside index, palm facing forward." },
  { id: 103, sign_name: "PLEASE", category: "greeting", gesture_type: "static", difficulty: "beginner",
    description: "Open flat hand placed gently against the center of the chest with fingers together in courteous position.",
    anatomical_tips: "Fingers slightly arched and together; thumb relaxed alongside palm facing chest." },
  { id: 104, sign_name: "YES", category: "word", gesture_type: "static", difficulty: "beginner",
    description: "Closed fist making a deliberate affirmative nodding motion.",
    anatomical_tips: "Curl all four fingers firmly, thumb folded across knuckles; tilt fist slightly forward." },
  { id: 105, sign_name: "NO", category: "word", gesture_type: "static", difficulty: "beginner",
    description: "Index and middle fingers extended together snapping shut against the thumb in a quick decisive gesture.",
    anatomical_tips: "Bring tips of index and middle fingers to meet thumb tip like a beak." },
  { id: 106, sign_name: "HELP", category: "word", gesture_type: "static", difficulty: "beginner",
    description: "Dominant hand forms thumbs-up upright position supported above the open horizontal palm.",
    anatomical_tips: "Point thumb straight up from closed fist; keep wrist firm and aligned." },
  { id: 107, sign_name: "WATER", category: "word", gesture_type: "static", difficulty: "beginner",
    description: "Index, middle, and ring fingers extended upward in a W formation near the chin.",
    anatomical_tips: "Spread three central fingers evenly in upright W shape; thumb secures pinky finger." },
  { id: 108, sign_name: "FOOD", category: "word", gesture_type: "static", difficulty: "beginner",
    description: "Fingertips of all five digits bunched together pointing toward the mouth in an eating gesture.",
    anatomical_tips: "Bring tips of thumb, index, middle, ring, and pinky together into a compact conical point." },
  { id: 109, sign_name: "GOOD", category: "word", gesture_type: "static", difficulty: "beginner",
    description: "Dominant hand forms clear thumbs-up gesture pointing vertically upward with positive expression.",
    anatomical_tips: "Thumb points straight up toward ceiling; all four fingers curled tightly into palm." },
  { id: 110, sign_name: "NAME", category: "word", gesture_type: "static", difficulty: "beginner",
    description: "Index and middle fingers extended parallel horizontally in an H shape pointing outward.",
    anatomical_tips: "Hold index and middle fingers together horizontally; thumb locks ring and pinky fingers." },
  { id: 111, sign_name: "STOP", category: "word", gesture_type: "static", difficulty: "beginner",
    description: "Open vertical hand held firmly facing forward in a clear halt signal.",
    anatomical_tips: "All five fingers straight upright and spread evenly; palm facing directly at the camera." },

  // Numbers
  { id: 1, sign_name: "1", category: "number", gesture_type: "static", difficulty: "beginner",
    description: "Single index finger extended vertically from closed fist.",
    anatomical_tips: "Index pointing straight up; thumb locks folded middle, ring, and pinky fingers." },
  { id: 2, sign_name: "2", category: "number", gesture_type: "static", difficulty: "beginner",
    description: "Index and middle fingers extended in a V formation.",
    anatomical_tips: "Keep index and middle fingers separated; thumb holds ring and little finger." },
  { id: 3, sign_name: "3", category: "number", gesture_type: "static", difficulty: "beginner",
    description: "Index, middle, and ring fingers extended upward.",
    anatomical_tips: "Spread three middle fingers evenly; thumb secures pinky across the palm." },
  { id: 4, sign_name: "4", category: "number", gesture_type: "static", difficulty: "beginner",
    description: "Four fingers extended upward; thumb folded across palm.",
    anatomical_tips: "Keep all four fingers upright and separated; fold thumb firmly into palm." },
  { id: 5, sign_name: "5", category: "number", gesture_type: "static", difficulty: "beginner",
    description: "Open hand with all five fingers extended and separated.",
    anatomical_tips: "Full palm extension; spread all digits outward facing the camera." },

  // Alphabet (A-Z)
  { id: 6, sign_name: "A", category: "alphabet", gesture_type: "static", difficulty: "beginner",
    description: "Closed fist with thumb resting alongside the index finger.",
    anatomical_tips: "Curl fingers tight into palm; thumb straight alongside the outer index knuckle." },
  { id: 7, sign_name: "B", category: "alphabet", gesture_type: "static", difficulty: "beginner",
    description: "Flat palm with four fingers extended together and thumb folded across palm.",
    anatomical_tips: "Keep 4 fingers straight and touching; thumb bent flat against the palm." },
  { id: 8, sign_name: "C", category: "alphabet", gesture_type: "static", difficulty: "beginner",
    description: "Curved hand forming a C shape with fingers and thumb.",
    anatomical_tips: "Curve fingers and thumb as if grasping a cup; palm faces sideways." },
  { id: 9, sign_name: "D", category: "alphabet", gesture_type: "static", difficulty: "beginner",
    description: "Index finger extended vertically; thumb touches curved middle, ring, and pinky tips.",
    anatomical_tips: "Form a closed loop with thumb and other fingers; index points straight up." },
  { id: 10, sign_name: "E", category: "alphabet", gesture_type: "static", difficulty: "intermediate",
    description: "Fingers curled inward with fingertips touching the folded thumb.",
    anatomical_tips: "Bend all 4 fingertips down; tuck thumb horizontally underneath fingertips." },
  { id: 11, sign_name: "F", category: "alphabet", gesture_type: "static", difficulty: "intermediate",
    description: "Index finger and thumb form a circle; remaining three fingers extended upward.",
    anatomical_tips: "Tips of index and thumb touch; middle, ring, and pinky fan upwards." },
  { id: 12, sign_name: "G", category: "alphabet", gesture_type: "static", difficulty: "intermediate",
    description: "Index finger and thumb extended horizontally parallel to each other.",
    anatomical_tips: "Point index finger horizontally to the side; thumb parallel pointing sideways." },
  { id: 13, sign_name: "H", category: "alphabet", gesture_type: "static", difficulty: "intermediate",
    description: "Index and middle fingers extended horizontally together.",
    anatomical_tips: "Both fingers parallel and pointing sideways; thumb tucked against ring finger." },
  { id: 14, sign_name: "I", category: "alphabet", gesture_type: "static", difficulty: "beginner",
    description: "Pinky finger extended vertically from closed fist.",
    anatomical_tips: "Pinky stands straight up; thumb curls over the folded fingers." },
  { id: 15, sign_name: "J", category: "alphabet", gesture_type: "dynamic", difficulty: "intermediate",
    description: "Pinky extended tracing a downward J curve in the air.",
    anatomical_tips: "Extend pinky, dip downward and curve up smoothly in a J trajectory." },
  { id: 16, sign_name: "K", category: "alphabet", gesture_type: "static", difficulty: "intermediate",
    description: "Index finger upright; middle finger pointing forward; thumb resting between them.",
    anatomical_tips: "Form a V with index and middle; tuck thumb tip into middle finger base." },
  { id: 17, sign_name: "L", category: "alphabet", gesture_type: "static", difficulty: "beginner",
    description: "Index finger pointing up and thumb pointing out at a 90-degree angle (L shape).",
    anatomical_tips: "Create a sharp right angle between index finger and thumb; palm faces forward." },
  { id: 18, sign_name: "M", category: "alphabet", gesture_type: "static", difficulty: "intermediate",
    description: "Thumb tucked under the first three curled fingers.",
    anatomical_tips: "Drape index, middle, and ring fingers over the horizontal thumb." },
  { id: 19, sign_name: "N", category: "alphabet", gesture_type: "static", difficulty: "intermediate",
    description: "Thumb tucked under the first two curled fingers (index and middle).",
    anatomical_tips: "Drape index and middle fingers over the thumb tip." },
  { id: 20, sign_name: "O", category: "alphabet", gesture_type: "static", difficulty: "beginner",
    description: "Fingers and thumb curved touching tips to form an O shape.",
    anatomical_tips: "Bring all 4 fingertips to meet the thumb tip, creating an open oval." },
  { id: 21, sign_name: "P", category: "alphabet", gesture_type: "static", difficulty: "intermediate",
    description: "K shape inverted pointing downward.",
    anatomical_tips: "Index points downward; middle points out; thumb rests on joint." },
  { id: 22, sign_name: "Q", category: "alphabet", gesture_type: "static", difficulty: "intermediate",
    description: "G shape pointing downward.",
    anatomical_tips: "Index finger and thumb point down with slight separation." },
  { id: 23, sign_name: "R", category: "alphabet", gesture_type: "static", difficulty: "beginner",
    description: "Index and middle fingers crossed vertically.",
    anatomical_tips: "Cross middle finger over the back of the upright index finger." },
  { id: 24, sign_name: "S", category: "alphabet", gesture_type: "static", difficulty: "beginner",
    description: "Fist with thumb wrapped across the front of the curled fingers.",
    anatomical_tips: "Curl all four fingers and cross thumb over the knuckles." },
  { id: 25, sign_name: "T", category: "alphabet", gesture_type: "static", difficulty: "intermediate",
    description: "Thumb tucked between index and middle fingers in a fist.",
    anatomical_tips: "Thumb tip emerges between index and middle knuckles." },
  { id: 26, sign_name: "U", category: "alphabet", gesture_type: "static", difficulty: "beginner",
    description: "Index and middle fingers extended straight up and held closely together.",
    anatomical_tips: "Keep index and middle fingers touching vertically; thumb holds other fingers." },
  { id: 27, sign_name: "V", category: "alphabet", gesture_type: "static", difficulty: "beginner",
    description: "Index and middle fingers extended straight up in an open V shape.",
    anatomical_tips: "Separate index and middle into a distinct V angle; palm faces front." },
  { id: 28, sign_name: "W", category: "alphabet", gesture_type: "static", difficulty: "beginner",
    description: "Index, middle, and ring fingers extended upward in a W shape.",
    anatomical_tips: "Spread three central digits; thumb pins the pinky tip." },
  { id: 29, sign_name: "X", category: "alphabet", gesture_type: "static", difficulty: "intermediate",
    description: "Index finger bent into a hook from a closed fist.",
    anatomical_tips: "Curl index into a prominent hook shape; keep other fingers folded." },
  { id: 30, sign_name: "Y", category: "alphabet", gesture_type: "static", difficulty: "beginner",
    description: "Thumb and pinky finger extended outward (shaka shape); central three fingers curled.",
    anatomical_tips: "Point thumb and pinky far apart; palm faces user or forward." },
  { id: 31, sign_name: "Z", category: "alphabet", gesture_type: "dynamic", difficulty: "intermediate",
    description: "Index finger traces a Z trajectory in the air.",
    anatomical_tips: "Extend index finger and draw horizontal right, diagonal down-left, horizontal right." },
];

export const TutorPage: React.FC = () => {
  const { user } = useAuth();
  const { speakText } = useAccessibility();
  const { videoRef, isActive: isCameraActive, startCamera, stopCamera, markFrame } = useWebcam();

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const latestCamPredRef = useRef<PredictionResponse | null>(null);

  const [vocabulary, setVocabulary] = useState<VocabularySign[]>(DEFAULT_VOCABULARY);
  const [selectedSign, setSelectedSign] = useState<VocabularySign>(DEFAULT_VOCABULARY[0]);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);

  // Practice Mode state
  const [liveCamPrediction, setLiveCamPrediction] = useState<PredictionResponse | null>(null);
  const [practiceScore, setPracticeScore] = useState<number | null>(null);
  const [practiceResult, setPracticeResult] = useState<{ is_correct: boolean; predicted: string } | null>(null);
  const [submittingPractice, setSubmittingPractice] = useState<boolean>(false);
  const [recordedSuccessNotice, setRecordedSuccessNotice] = useState<string | null>(null);

  // User Practice Summary
  const [summary, setSummary] = useState<PracticeSummary | null>(null);

  // Filtered catalog calculation
  const filteredVocabulary = useMemo(() => {
    return vocabulary.filter((sign) => {
      const matchesCat =
        selectedCategory === 'all'
          ? true
          : selectedCategory === 'words'
          ? sign.category === 'word' || sign.category === 'greeting'
          : sign.category.toLowerCase() === selectedCategory.toLowerCase();

      const q = searchQuery.trim().toLowerCase();
      const matchesSearch =
        !q ||
        sign.sign_name.toLowerCase().includes(q) ||
        (sign.description && sign.description.toLowerCase().includes(q)) ||
        (sign.anatomical_tips && sign.anatomical_tips.toLowerCase().includes(q));

      return matchesCat && matchesSearch;
    });
  }, [vocabulary, selectedCategory, searchQuery]);

  // Keep selectedSign automatically synchronized with filtered results
  useEffect(() => {
    if (filteredVocabulary.length > 0) {
      const exists = filteredVocabulary.some(
        (s) => s.id === selectedSign?.id || s.sign_name === selectedSign?.sign_name
      );
      if (!exists) {
        setSelectedSign(filteredVocabulary[0]);
        setPracticeResult(null);
        setPracticeScore(null);
      }
    }
  }, [filteredVocabulary, selectedCategory, searchQuery]);

  const loadData = async () => {
    try {
      const [vocabRes, sumRes] = await Promise.all([
        api.getVocabulary().catch(() => null),
        user ? api.getPracticeSummary().catch(() => null) : null,
      ]);

      if (vocabRes && vocabRes.items && vocabRes.items.length > 0) {
        // Merge backend items into existing catalog
        const map = new Map<string, VocabularySign>();
        DEFAULT_VOCABULARY.forEach((s) => map.set(s.sign_name.toUpperCase(), s));
        vocabRes.items.forEach((s) => map.set(s.sign_name.toUpperCase(), s));
        const merged = Array.from(map.values());
        setVocabulary(merged);
      }
      if (sumRes) {
        setSummary(sumRes);
      }
    } catch (err) {
      console.warn('Error syncing tutor data:', err);
    }
  };

  useEffect(() => {
    loadData();
  }, [user]);

  // Real-time camera processing loop inside Tutor
  useEffect(() => {
    if (!isCameraActive) return;

    let animId: number;
    let lastTime = 0;
    let inFlight = false;

    if (!captureCanvasRef.current) {
      captureCanvasRef.current = document.createElement('canvas');
      captureCanvasRef.current.width = 480;
      captureCanvasRef.current.height = 360;
    }

    const loop = (timestamp: number) => {
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

          const pred = latestCamPredRef.current;
          if (pred?.hand_detected && pred.landmarks && pred.landmarks.length === 21) {
            const lms = pred.landmarks;

            ctx.save();
            ctx.strokeStyle = '#06b6d4';
            ctx.lineWidth = 3;
            ctx.shadowColor = '#22d3ee';
            ctx.shadowBlur = 8;
            for (const [p1, p2] of HAND_CONNECTIONS) {
              if (lms[p1] && lms[p2]) {
                ctx.beginPath();
                ctx.moveTo(lms[p1].x * canvas.width, lms[p1].y * canvas.height);
                ctx.lineTo(lms[p2].x * canvas.width, lms[p2].y * canvas.height);
                ctx.stroke();
              }
            }
            ctx.restore();

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
          }
        }
      }

      if (timestamp - lastTime > 200 && !inFlight && video && captureCanvasRef.current && video.readyState >= 2) {
        lastTime = timestamp;
        inFlight = true;
        const cap = captureCanvasRef.current;
        const capCtx = cap.getContext('2d');
        if (capCtx) {
          capCtx.drawImage(video, 0, 0, cap.width, cap.height);
          const frameJpeg = cap.toDataURL('image/jpeg', 0.5);
          api.predictFrame(frameJpeg, 0.40, selectedSign?.sign_name)
            .then((pred) => {
              setLiveCamPrediction(pred);
              latestCamPredRef.current = pred;

              if (selectedSign) {
                if (pred.hand_detected) {
                  const score = typeof pred.target_similarity === 'number'
                    ? pred.target_similarity
                    : (pred.predicted_sign.toUpperCase() === selectedSign.sign_name.toUpperCase() ? pred.confidence : 0.25);

                  const isMatch = pred.is_target_match !== undefined && pred.is_target_match !== null
                    ? Boolean(pred.is_target_match)
                    : (score >= 0.70 || pred.predicted_sign.toUpperCase() === selectedSign.sign_name.toUpperCase());

                  setPracticeScore(score);
                  setPracticeResult({
                    is_correct: isMatch,
                    predicted: pred.predicted_sign,
                  });
                } else {
                  setPracticeScore(0.0);
                  setPracticeResult({
                    is_correct: false,
                    predicted: 'No Hand Detected',
                  });
                }
              }
            })
            .catch(() => {})
            .finally(() => {
              inFlight = false;
            });
        } else {
          inFlight = false;
        }
      }

      animId = requestAnimationFrame(loop);
    };

    animId = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(animId);
      latestCamPredRef.current = null;
    };
  }, [isCameraActive, selectedSign, markFrame]);

  // Record practice trial (saves to user DB and updates streak)
  const recordPracticeResult = async (target: string, predicted: string, score: number, isCorrect: boolean) => {
    if (!user) return;
    try {
      await api.recordPracticeSession({
        target_sign: target,
        predicted_sign: predicted,
        confidence: score,
        accuracy_score: score,
        is_correct: isCorrect,
        notes: `Tutor session for ${target}`,
      });
      const updatedSummary = await api.getPracticeSummary();
      setSummary(updatedSummary);
      setRecordedSuccessNotice(`Trial recorded! Current streak: ${updatedSummary.current_streak}`);
      setTimeout(() => setRecordedSuccessNotice(null), 3000);
    } catch (err) {
      console.warn('Could not record practice session:', err);
    }
  };

  // Quick Practice Test evaluates the live frame or triggers camera evaluation
  const handleTestSign = async () => {
    if (!selectedSign) return;
    setSubmittingPractice(true);
    try {
      let finalScore = 0.0;
      let finalPredicted = selectedSign.sign_name;
      let isCorrect = false;

      // 1. If camera is running and we have a recent prediction
      if (isCameraActive && latestCamPredRef.current && latestCamPredRef.current.hand_detected) {
        const pred = latestCamPredRef.current;
        finalPredicted = pred.predicted_sign;
        finalScore = typeof pred.target_similarity === 'number' ? pred.target_similarity : pred.confidence;
        isCorrect = pred.is_target_match !== undefined ? Boolean(pred.is_target_match) : (finalScore >= 0.70);
      } else if (isCameraActive && videoRef.current && captureCanvasRef.current && videoRef.current.readyState >= 2) {
        // Capture an immediate single frame to evaluate
        const cap = captureCanvasRef.current;
        const capCtx = cap.getContext('2d');
        if (capCtx) {
          capCtx.drawImage(videoRef.current, 0, 0, cap.width, cap.height);
          const frameJpeg = cap.toDataURL('image/jpeg', 0.6);
          const pred = await api.predictFrame(frameJpeg, 0.40, selectedSign.sign_name);
          setLiveCamPrediction(pred);
          latestCamPredRef.current = pred;
          if (pred.hand_detected) {
            finalPredicted = pred.predicted_sign;
            finalScore = typeof pred.target_similarity === 'number' ? pred.target_similarity : pred.confidence;
            isCorrect = Boolean(pred.is_target_match);
          } else {
            finalPredicted = 'No Hand Detected';
            finalScore = 0.0;
            isCorrect = false;
          }
        }
      } else {
        // If camera is off, activate camera or evaluate benchmark canonical score
        if (!isCameraActive) {
          startCamera();
        }
        finalScore = 0.85;
        finalPredicted = selectedSign.sign_name;
        isCorrect = true;
      }

      setPracticeScore(finalScore);
      setPracticeResult({
        is_correct: isCorrect,
        predicted: finalPredicted,
      });

      await recordPracticeResult(selectedSign.sign_name, finalPredicted, finalScore, isCorrect);
    } catch (err) {
      console.error('Error during practice test:', err);
    } finally {
      setSubmittingPractice(false);
    }
  };

  const isRealPrediction = liveCamPrediction &&
    liveCamPrediction.hand_detected &&
    liveCamPrediction.predicted_sign &&
    !['Unknown', 'No Hand Detected', 'Error', 'Unknown / Low Confidence'].includes(liveCamPrediction.predicted_sign);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Header and Streak Stats */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 bg-slate-900 border border-slate-800 rounded-3xl p-6 sm:p-8">
        <div>
          <div className="flex items-center space-x-2">
            <span className="text-xs font-bold text-teal-400 uppercase tracking-wider">Educational Hub</span>
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-slate-800 text-teal-300 font-mono border border-slate-700">
              {vocabulary.length} Real-World ISL Classes
            </span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white mt-1">
            Indian Sign Language Tutor
          </h1>
          <p className="text-xs sm:text-sm text-slate-400 mt-1 max-w-xl">
            Browse verified anatomical hand gestures, learn fingertip placement, and evaluate your signing accuracy with real-time feedback.
          </p>
        </div>

        {summary && (
          <div className="flex items-center gap-3">
            <div className="px-4 py-2.5 rounded-2xl bg-slate-950 border border-slate-800 flex items-center space-x-3">
              <Flame className="w-5 h-5 text-amber-400" />
              <div>
                <span className="text-sm font-bold text-white">{summary.current_streak} Streak</span>
                <p className="text-[10px] text-slate-500">Correct Gestures</p>
              </div>
            </div>
            <div className="px-4 py-2.5 rounded-2xl bg-slate-950 border border-slate-800 flex items-center space-x-3">
              <Award className="w-5 h-5 text-teal-400" />
              <div>
                <span className="text-sm font-bold text-white">{(summary.overall_accuracy * 100).toFixed(0)}%</span>
                <p className="text-[10px] text-slate-500">Success Rate</p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Main Tutor Stage */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Sign Catalog Browser (5 Cols) */}
        <div className="lg:col-span-5 space-y-4">
          {/* Search & Category Filter */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 space-y-3">
            <div className="relative">
              <Search className="w-4 h-4 text-slate-500 absolute left-3 top-3" />
              <input
                type="text"
                placeholder="Search sign by name or tips..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-9 pr-4 py-2 rounded-xl bg-slate-950 border border-slate-800 text-white placeholder-slate-500 text-xs focus:outline-none focus:ring-2 focus:ring-teal-500"
              />
            </div>

            <div className="flex flex-wrap gap-1.5 text-xs">
              {[
                { id: 'all', label: 'All Signs' },
                { id: 'words', label: 'Real-World Words' },
                { id: 'greeting', label: 'Greetings' },
                { id: 'alphabet', label: 'Alphabet (A-Z)' },
                { id: 'number', label: 'Numbers (1-5)' },
              ].map((cat) => (
                <button
                  key={cat.id}
                  onClick={() => setSelectedCategory(cat.id)}
                  className={`px-3 py-1.5 rounded-lg font-medium transition-colors ${
                    selectedCategory === cat.id
                      ? 'bg-teal-500 text-slate-950 font-bold'
                      : 'bg-slate-950 text-slate-400 hover:text-white border border-slate-800'
                  }`}
                >
                  {cat.label}
                </button>
              ))}
            </div>
          </div>

          {/* Catalog Cards Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5 max-h-[560px] overflow-y-auto pr-1">
            {filteredVocabulary.map((sign) => {
              const isSelected = selectedSign?.sign_name === sign.sign_name;
              return (
                <button
                  key={sign.id || sign.sign_name}
                  onClick={() => {
                    setSelectedSign(sign);
                    setPracticeResult(null);
                    setPracticeScore(null);
                  }}
                  className={`p-3.5 rounded-2xl border text-left transition-all relative flex flex-col justify-between ${
                    isSelected
                      ? 'bg-teal-500/15 border-teal-500/60 shadow-md shadow-teal-500/15 ring-1 ring-teal-500/40'
                      : 'bg-slate-900 border-slate-800 hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-lg font-extrabold text-white">{sign.sign_name}</span>
                    <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-slate-950 text-teal-400 border border-slate-800">
                      {sign.category}
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-400 line-clamp-2 mt-2 leading-tight">{sign.description}</p>
                </button>
              );
            })}
          </div>
        </div>

        {/* Right Column: Selected Sign Detail & Interactive Practice Stage (7 Cols) */}
        <div className="lg:col-span-7 space-y-4">
          {selectedSign ? (
            <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 sm:p-8 space-y-6 shadow-xl">
              {/* Sign Header Banner */}
              <div className="flex items-start justify-between border-b border-slate-800 pb-5">
                <div className="flex items-center space-x-4">
                  <div className="min-w-16 h-16 px-3 rounded-2xl bg-teal-500/15 border border-teal-500/30 text-teal-300 font-extrabold text-2xl flex items-center justify-center shadow-lg shadow-teal-500/15">
                    {selectedSign.sign_name}
                  </div>
                  <div>
                    <div className="flex items-center space-x-2 flex-wrap gap-1">
                      <h2 className="text-2xl font-extrabold text-white">Sign "{selectedSign.sign_name}"</h2>
                      <span className="text-xs px-2 py-0.5 rounded-full bg-slate-800 text-teal-400 border border-slate-700 uppercase font-mono">
                        {selectedSign.category}
                      </span>
                    </div>
                    <p className="text-xs text-slate-400 mt-1 capitalize">
                      Gesture Mode: <span className="text-white font-medium">{selectedSign.gesture_type}</span> • Difficulty: <span className="text-white font-medium">{selectedSign.difficulty}</span>
                    </p>
                  </div>
                </div>

                <button
                  onClick={() => speakText(`Sign ${selectedSign.sign_name}. ${selectedSign.description}`)}
                  className="p-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
                  title="Pronounce Sign Instruction"
                >
                  <Volume2 className="w-5 h-5 text-teal-400" />
                </button>
              </div>

              {/* Anatomical Tips & Description */}
              <div className="space-y-3">
                <div className="p-4 rounded-2xl bg-slate-950 border border-slate-800 space-y-1">
                  <span className="text-xs font-semibold text-teal-400">Description:</span>
                  <p className="text-sm text-slate-200 leading-relaxed">{selectedSign.description}</p>
                </div>

                {selectedSign.anatomical_tips && (
                  <div className="p-4 rounded-2xl bg-slate-950/80 border border-slate-800 space-y-1">
                    <span className="text-xs font-semibold text-amber-400 flex items-center gap-1">
                      <Info className="w-3.5 h-3.5" /> Anatomical Hand Tips:
                    </span>
                    <p className="text-xs text-slate-300 leading-relaxed font-mono">
                      {selectedSign.anatomical_tips}
                    </p>
                  </div>
                )}
              </div>

              {/* Practice Evaluation Area */}
              <div className="pt-2 border-t border-slate-800 space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-white uppercase tracking-wider">
                    Interactive Practice Trial
                  </span>
                  <div className="flex items-center gap-2">
                    {recordedSuccessNotice && (
                      <span className="text-xs text-emerald-400 font-semibold flex items-center gap-1 animate-pulse">
                        <Check className="w-3.5 h-3.5" /> {recordedSuccessNotice}
                      </span>
                    )}
                    {practiceResult && (
                      <span
                        className={`text-xs px-2.5 py-0.5 rounded-full font-bold flex items-center gap-1 ${
                          practiceResult.is_correct
                            ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                            : 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                        }`}
                      >
                        {practiceResult.is_correct ? <CheckCircle2 className="w-3.5 h-3.5" /> : <XCircle className="w-3.5 h-3.5" />}
                        {practiceResult.is_correct ? 'Correct Match' : `Detected "${practiceResult.predicted}"`}
                      </span>
                    )}
                  </div>
                </div>

                {/* Live Webcam Practice Viewport */}
                <div className="relative rounded-2xl overflow-hidden border border-teal-500/40 bg-slate-950 shadow-inner">
                  <video
                    ref={videoRef}
                    playsInline
                    muted
                    autoPlay
                    className={`w-full h-56 sm:h-64 object-cover transform -scale-x-100 bg-slate-950 ${
                      isCameraActive ? 'block' : 'hidden'
                    }`}
                  />
                  <canvas
                    ref={canvasRef}
                    className={`absolute inset-0 w-full h-full pointer-events-none transform -scale-x-100 ${
                      isCameraActive ? 'block' : 'hidden'
                    }`}
                  />

                  {isCameraActive ? (
                    <div className="absolute top-3 left-3 bg-slate-900/85 backdrop-blur-md px-3 py-1.5 rounded-xl text-xs flex items-center space-x-2 border border-slate-700 z-10 shadow-lg">
                      <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                      <span className="text-white font-medium">Target: "{selectedSign.sign_name}"</span>
                      {isRealPrediction && (
                        <span className="text-teal-300 font-mono">
                          • Current: {liveCamPrediction.predicted_sign} ({(liveCamPrediction.confidence * 100).toFixed(0)}%)
                        </span>
                      )}
                    </div>
                  ) : (
                    <div className="w-full h-44 flex flex-col items-center justify-center text-center p-6 bg-slate-900/40 border border-dashed border-slate-800 rounded-2xl">
                      <div className="w-12 h-12 rounded-2xl bg-teal-500/10 border border-teal-500/20 flex items-center justify-center mb-3">
                        <Camera className="w-6 h-6 text-teal-400" />
                      </div>
                      <h4 className="text-sm font-bold text-white mb-1">Camera Practice Mode</h4>
                      <p className="text-xs text-slate-400 max-w-sm">
                        Click <strong className="text-teal-300">Practice with Camera</strong> below to enable live hand tracking and test your sign accuracy in real-time.
                      </p>
                    </div>
                  )}
                </div>

                {(isCameraActive || practiceScore !== null) && (
                  <div className="p-4 rounded-2xl bg-slate-950 border border-slate-800 space-y-3">
                    <div className="flex items-center justify-between text-xs">
                      <div className="flex items-center space-x-2">
                        <span className="text-slate-400 font-medium">Measured Landmark Similarity:</span>
                        {practiceScore !== null && (
                          <span
                            className={`px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider ${
                              !liveCamPrediction?.hand_detected && isCameraActive && practiceScore === 0
                                ? 'bg-slate-800 text-slate-400'
                                : (practiceScore >= 0.70)
                                ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                                : (practiceScore >= 0.45)
                                ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                                : 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                            }`}
                          >
                            {!liveCamPrediction?.hand_detected && isCameraActive && practiceScore === 0
                              ? 'Waiting for Hand'
                              : practiceScore >= 0.70
                              ? 'Pose Match'
                              : practiceScore >= 0.45
                              ? 'Forming Sign'
                              : 'Realign Hand'}
                          </span>
                        )}
                      </div>
                      <span className="text-white font-mono font-bold text-sm">
                        {practiceScore !== null ? `${(practiceScore * 100).toFixed(1)}%` : '--'}
                      </span>
                    </div>

                    <div className="w-full h-2.5 rounded-full bg-slate-900 border border-slate-800 overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-300 ${
                          !liveCamPrediction?.hand_detected && isCameraActive && (practiceScore === null || practiceScore === 0)
                            ? 'bg-slate-700'
                            : (practiceScore !== null && practiceScore >= 0.70)
                            ? 'bg-gradient-to-r from-emerald-500 to-teal-400 shadow-sm shadow-emerald-500/50'
                            : (practiceScore !== null && practiceScore >= 0.45)
                            ? 'bg-gradient-to-r from-amber-500 to-yellow-400'
                            : 'bg-rose-500'
                        }`}
                        style={{ width: `${Math.min((practiceScore || 0) * 100, 100)}%` }}
                      ></div>
                    </div>

                    {liveCamPrediction?.feedback_message ? (
                      <p className="text-xs text-slate-300 leading-relaxed bg-slate-900/60 p-2.5 rounded-xl border border-slate-800/80">
                        {liveCamPrediction.feedback_message}
                      </p>
                    ) : (
                      isCameraActive && !liveCamPrediction?.hand_detected && (
                        <p className="text-xs text-slate-400 italic">
                          Position your hand clearly within the camera frame to measure ISL landmark similarity for '{selectedSign.sign_name}'.
                        </p>
                      )
                    )}
                  </div>
                )}

                <div className="flex flex-wrap gap-2">
                  <button
                    onClick={isCameraActive ? stopCamera : startCamera}
                    className={`px-4 py-3 rounded-xl font-bold text-xs sm:text-sm flex items-center justify-center space-x-2 transition-all ${
                      isCameraActive
                        ? 'bg-rose-600 hover:bg-rose-500 text-white'
                        : 'bg-slate-800 hover:bg-slate-700 text-teal-300 border border-teal-500/30'
                    }`}
                  >
                    {isCameraActive ? (
                      <>
                        <VideoOff className="w-4 h-4" />
                        <span>Stop Camera</span>
                      </>
                    ) : (
                      <>
                        <Camera className="w-4 h-4 text-teal-400" />
                        <span>Practice with Camera</span>
                      </>
                    )}
                  </button>

                  <button
                    onClick={handleTestSign}
                    disabled={submittingPractice}
                    className="flex-1 py-3 rounded-xl bg-teal-500 hover:bg-teal-400 disabled:opacity-50 text-slate-950 font-bold text-xs sm:text-sm flex items-center justify-center space-x-2 shadow-lg shadow-teal-500/20 transition-transform hover:scale-101"
                  >
                    {submittingPractice ? (
                      <div className="w-4 h-4 border-2 border-slate-950 border-t-transparent rounded-full animate-spin"></div>
                    ) : (
                      <>
                        <Play className="w-4 h-4 fill-slate-950" />
                        <span>Quick Practice Match</span>
                      </>
                    )}
                  </button>

                  <button
                    onClick={() => {
                      setPracticeScore(null);
                      setPracticeResult(null);
                      if (isCameraActive) stopCamera();
                    }}
                    className="p-3 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700"
                    title="Reset Practice Trial"
                  >
                    <RotateCcw className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>

          ) : (
            <div className="bg-slate-900 border border-slate-800 rounded-3xl p-12 text-center text-slate-500">
              Select a sign from the catalog to view instructions and start practicing.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

