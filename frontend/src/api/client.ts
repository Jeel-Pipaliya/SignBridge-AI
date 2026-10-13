import {
  AuthResponse,
  PredictionResponse,
  GrammarCorrectResponse,
  PracticeSession,
  PracticeSummary,
  AnalyticsOverview,
  TranscriptRecord,
  User,
  UserPreferences,
  VocabularyListResponse,
  VocabularySign,
} from './types';

const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api/v1';

class ApiClient {
  private getHeaders(): HeadersInit {
    const token = localStorage.getItem('signbridge_token');
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }
    return headers;
  }

  private async handleResponse<T>(res: Response): Promise<T> {
    if (!res.ok) {
      let errorMsg = `API Error: ${res.status} ${res.statusText}`;
      try {
        const body = await res.json();
        if (body.detail) {
          errorMsg = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
        }
      } catch {
        // use default error message
      }
      throw new Error(errorMsg);
    }
    return res.json();
  }

  // Health
  async getHealth() {
    const res = await fetch(`${API_BASE}/health`);
    return this.handleResponse<{ status: string; version: string }>(res);
  }

  async getReady() {
    const res = await fetch(`${API_BASE}/ready`);
    return this.handleResponse<{ status: string; database: boolean; static_model: boolean }>(res);
  }

  // Auth
  async login(email: string, password: string): Promise<AuthResponse> {
    const res = await fetch(`${API_BASE}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    const data = await this.handleResponse<AuthResponse>(res);
    localStorage.setItem('signbridge_token', data.access_token);
    return data;
  }

  async register(email: string, password: string, full_name?: string): Promise<AuthResponse> {
    const res = await fetch(`${API_BASE}/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password, full_name }),
    });
    const data = await this.handleResponse<AuthResponse>(res);
    localStorage.setItem('signbridge_token', data.access_token);
    return data;
  }

  async getCurrentUser(): Promise<User> {
    const res = await fetch(`${API_BASE}/auth/me`, {
      headers: this.getHeaders(),
    });
    return this.handleResponse<User>(res);
  }

  logout() {
    localStorage.removeItem('signbridge_token');
  }

  // User Preferences
  async updatePreferences(prefs: Partial<UserPreferences>): Promise<UserPreferences> {
    const res = await fetch(`${API_BASE}/users/preferences`, {
      method: 'PUT',
      headers: this.getHeaders(),
      body: JSON.stringify(prefs),
    });
    return this.handleResponse<UserPreferences>(res);
  }

  // Vocabulary
  async getVocabulary(category?: string, search?: string): Promise<VocabularyListResponse> {
    const params = new URLSearchParams();
    if (category) params.append('category', category);
    if (search) params.append('search', search);
    const res = await fetch(`${API_BASE}/vocabulary?${params.toString()}`);
    return this.handleResponse<VocabularyListResponse>(res);
  }

  async getVocabularySign(id: number): Promise<VocabularySign> {
    const res = await fetch(`${API_BASE}/vocabulary/${id}`);
    return this.handleResponse<VocabularySign>(res);
  }

  // Model Info
  async getModelInfo() {
    const res = await fetch(`${API_BASE}/model/info`);
    return this.handleResponse<any>(res);
  }

  // Recognition
  async predictFrame(imageData: string, threshold?: number, targetSign?: string): Promise<PredictionResponse> {
    const res = await fetch(`${API_BASE}/recognition/predict-frame`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify({
        image: imageData,
        confidence_threshold: threshold,
        target_sign: targetSign,
      }),
    });
    return this.handleResponse<PredictionResponse>(res);
  }

  async predictLandmarks(landmarks: Array<{ x: number; y: number; z: number }>, threshold?: number, targetSign?: string): Promise<PredictionResponse> {
    const res = await fetch(`${API_BASE}/recognition/predict-landmarks`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify({
        landmarks,
        handedness: 'Right',
        confidence_threshold: threshold,
        target_sign: targetSign,
      }),
    });
    return this.handleResponse<PredictionResponse>(res);
  }

  async grammarCorrect(tokens: string[], target_language: string = 'en'): Promise<GrammarCorrectResponse> {
    const res = await fetch(`${API_BASE}/recognition/grammar-correct`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify({ tokens, target_language }),
    });
    return this.handleResponse<GrammarCorrectResponse>(res);
  }

  async submitFeedback(detected_sign: string, corrected_sign: string, confidence?: number) {
    const res = await fetch(`${API_BASE}/recognition/feedback`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify({ detected_sign, corrected_sign, confidence }),
    });
    return this.handleResponse<any>(res);
  }

  // Practice
  async recordPracticeSession(session: {
    target_sign: string;
    predicted_sign?: string;
    confidence?: number;
    accuracy_score: number;
    is_correct: boolean;
    notes?: string;
  }): Promise<PracticeSession> {
    const res = await fetch(`${API_BASE}/practice/sessions`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify(session),
    });
    return this.handleResponse<PracticeSession>(res);
  }

  async getPracticeSummary(): Promise<PracticeSummary> {
    const res = await fetch(`${API_BASE}/practice/summary`, {
      headers: this.getHeaders(),
    });
    return this.handleResponse<PracticeSummary>(res);
  }

  // Analytics
  async getAnalyticsOverview(): Promise<AnalyticsOverview> {
    const res = await fetch(`${API_BASE}/analytics/overview`, {
      headers: this.getHeaders(),
    });
    return this.handleResponse<AnalyticsOverview>(res);
  }

  // Transcripts
  async saveTranscript(transcript: { title: string; content: string; mode: string; language: string }): Promise<TranscriptRecord> {
    const res = await fetch(`${API_BASE}/transcripts`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify(transcript),
    });
    return this.handleResponse<TranscriptRecord>(res);
  }

  async listTranscripts(mode?: string): Promise<TranscriptRecord[]> {
    const params = new URLSearchParams();
    if (mode) params.append('mode', mode);
    const res = await fetch(`${API_BASE}/transcripts?${params.toString()}`, {
      headers: this.getHeaders(),
    });
    return this.handleResponse<TranscriptRecord[]>(res);
  }

  async clearHistory(clearPractice: boolean = false) {
    const res = await fetch(`${API_BASE}/history/clear?clear_practice=${clearPractice}`, {
      method: 'DELETE',
      headers: this.getHeaders(),
    });
    return this.handleResponse<any>(res);
  }
}

export const api = new ApiClient();
