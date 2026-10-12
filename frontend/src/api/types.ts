export interface User {
  id: number;
  email: string;
  full_name?: string;
  is_active: boolean;
  is_superuser: boolean;
  created_at: string;
  preferences?: UserPreferences;
}

export interface UserPreferences {
  id: number;
  user_id: number;
  theme: 'dark' | 'light';
  font_size: 'normal' | 'large';
  high_contrast: boolean;
  reduced_motion: boolean;
  tts_speed: number;
  tts_volume: number;
  preferred_language: 'en' | 'hi';
  auto_speak: boolean;
  store_history: boolean;
  updated_at: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface VocabularySign {
  id: number;
  sign_name: string;
  category: 'alphabet' | 'number' | 'greeting' | 'phrase';
  description?: string;
  gesture_type: 'static' | 'dynamic';
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  thumbnail_url?: string;
  anatomical_tips?: string;
  created_at: string;
}

export interface VocabularyListResponse {
  total: number;
  items: VocabularySign[];
}

export interface LandmarkPoint {
  x: number;
  y: number;
  z: number;
}

export interface CandidatePrediction {
  sign: string;
  confidence: number;
}

export interface PredictionResponse {
  predicted_sign: string;
  confidence: number;
  is_recognized: boolean;
  is_supported_vocabulary: boolean;
  model_name: string;
  model_version: string;
  top_candidates: CandidatePrediction[];
  inference_duration_ms: number;
  handedness?: string;
}

export interface GrammarCorrectResponse {
  raw_tokens: string[];
  raw_sentence: string;
  corrected_sentence: string;
  language: string;
  translated_sentence?: string;
}

export interface PracticeSession {
  id: number;
  user_id: number;
  target_sign: string;
  predicted_sign?: string;
  confidence?: number;
  accuracy_score: number;
  is_correct: boolean;
  notes?: string;
  created_at: string;
}

export interface PracticeSummary {
  total_sessions: number;
  correct_count: number;
  overall_accuracy: number;
  current_streak: number;
  most_practiced_signs: Array<{ sign: string; count: number }>;
}

export interface VocabularyBreakdownItem {
  sign: string;
  practice_count: number;
  success_rate: number;
  category: string;
}

export interface ModelBenchmarks {
  static_model: {
    available: boolean;
    model_name: string;
    version: string;
    architecture: string;
    num_classes: number;
    classes: string[];
    confidence_threshold: number;
    train_accuracy: number;
    val_accuracy: number;
    test_accuracy: number;
    test_macro_f1: number;
  };
  dynamic_model: {
    available: boolean;
    model_name: string;
    architecture: string;
    classes: string[];
    sequence_length: number;
    test_accuracy: number;
    test_macro_f1: number;
  };
}

export interface AnalyticsOverview {
  total_recognitions: number;
  total_practice_sessions: number;
  average_practice_accuracy: number;
  most_practiced_signs: VocabularyBreakdownItem[];
  recent_activity_timeline: Array<{
    id: number;
    type: string;
    target: string;
    accuracy: number;
    is_correct: boolean;
    timestamp: string;
  }>;
  model_benchmarks: ModelBenchmarks;
  offline_vs_live_note: string;
}

export interface TranscriptRecord {
  id: number;
  user_id?: number;
  title: string;
  content: string;
  mode: 'conversation' | 'meeting' | 'emergency';
  language: string;
  created_at: string;
}
