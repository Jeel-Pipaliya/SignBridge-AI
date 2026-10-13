import React, { useState, useEffect } from 'react';
import {
  MessageSquare,
  Mic,
  MicOff,
  Volume2,
  Copy,
  Check,
  Trash2,
  Download,
  Languages,
  Send,
  Save,
  Clock,
  Sparkles,
} from 'lucide-react';
import { useAccessibility } from '../context/AccessibilityContext';
import { api } from '../api/client';
import { TranscriptRecord } from '../api/types';

interface ChatMessage {
  id: string;
  sender: 'deaf_user' | 'hearing_user';
  text: string;
  translatedText?: string;
  timestamp: string;
}

export const CommunicationPage: React.FC = () => {
  const { speakText } = useAccessibility();

  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: '1',
      sender: 'hearing_user',
      text: 'Hello! How can I assist you today?',
      translatedText: 'नमस्ते! मैं आज आपकी क्या मदद कर सकता हूँ?',
      timestamp: '10:00 AM',
    },
    {
      id: '2',
      sender: 'deaf_user',
      text: 'I want to drink water.',
      translatedText: 'मुझे पानी पीना है।',
      timestamp: '10:01 AM',
    },
  ]);

  const [inputText, setInputText] = useState<string>('');
  const [activeSender, setActiveSender] = useState<'deaf_user' | 'hearing_user'>('deaf_user');
  const [selectedLanguage, setSelectedLanguage] = useState<string>('hi');
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [speechError, setSpeechError] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [saveStatus, setSaveStatus] = useState<string | null>(null);

  const QUICK_PHRASES = [
    { text: "Hello! How can I assist you today?", sender: 'hearing_user' as const },
    { text: "I want to drink water.", sender: 'deaf_user' as const },
    { text: "Where is the consultation room?", sender: 'deaf_user' as const },
    { text: "Please wait here, the doctor will see you shortly.", sender: 'hearing_user' as const },
    { text: "Thank you for understanding.", sender: 'deaf_user' as const },
  ];

  const CLIENT_HINDI_DICTIONARY: Record<string, string> = {
    "hello! how can i assist you today?": "नमस्ते! मैं आज आपकी क्या मदद कर सकता हूँ?",
    "i want to drink water.": "मुझे पानी पीना है।",
    "where is the consultation room?": "परामर्श कक्ष कहाँ है?",
    "please wait here, the doctor will see you shortly.": "कृपया यहाँ प्रतीक्षा करें, डॉक्टर जल्द ही आपसे मिलेंगे।",
    "thank you for understanding.": "समझने के लिए धन्यवाद।",
    "good morning, please take a seat.": "सुप्रभात, कृपया बैठिए।",
    "can you tell me what symptoms you are experiencing?": "क्या आप बता सकते हैं कि आपको क्या लक्षण महसूस हो रहे हैं?",
    "the pharmacy is on the ground floor to your left.": "दवाखाना भूतल पर आपके बाईं ओर है।",
    "is there anyone accompanying you today?": "क्या आज आपके साथ कोई आया है?",
    "what is your name?": "आपका नाम क्या है?",
    "my name is": "मेरा नाम है",
    "i need help": "मुझे मदद चाहिए",
    "call the doctor": "डॉक्टर को बुलाओ",
    "i have pain": "मुझे दर्द हो रहा है",
    "where is the hospital?": "अस्पताल कहाँ है?",
    "thank you": "धन्यवाद",
    "thanks": "धन्यवाद",
    "please": "कृपया",
    "yes": "हाँ",
    "no": "नहीं",
    "help": "मदद",
    "water": "पानी",
    "food": "खाना / भोजन",
    "good": "अच्छा",
    "name": "नाम",
    "stop": "रुको",
    "hello": "नमस्ते",
    "hi": "नमस्ते",
    "doctor": "डॉक्टर",
    "hospital": "अस्पताल",
    "medicine": "दवाई",
    "pain": "दर्द",
    "fever": "बुखार",
    "emergency": "आपातकाल",
  };

  const getClientHindiFallback = (text: string): string | undefined => {
    const clean = text.trim().toLowerCase();
    if (CLIENT_HINDI_DICTIONARY[clean]) return CLIENT_HINDI_DICTIONARY[clean];
    const stripped = clean.replace(/[.,?!]/g, '').trim();
    if (CLIENT_HINDI_DICTIONARY[stripped]) return CLIENT_HINDI_DICTIONARY[stripped];
    return undefined;
  };

  // Web Speech Recognition setup with simulation fallback
  const handleToggleMic = () => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setSpeechError('Web Speech API is not supported in this browser. You can type directly or click any quick phrase below.');
      return;
    }

    if (isRecording) {
      setIsRecording(false);
      return;
    }

    try {
      const recognition = new SpeechRecognition();
      recognition.lang = 'en-US';
      recognition.continuous = false;
      recognition.interimResults = false;

      recognition.onstart = () => {
        setIsRecording(true);
        setSpeechError(null);
      };

      recognition.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript;
        setInputText(transcript);
        setActiveSender('hearing_user');
        setIsRecording(false);
      };

      recognition.onerror = (event: any) => {
        setSpeechError(`Microphone notice: ${event.error}. You can still type or click quick phrases to test dialogue.`);
        setIsRecording(false);
      };

      recognition.onend = () => {
        setIsRecording(false);
      };

      recognition.start();
    } catch (err: any) {
      setSpeechError(`Could not access microphone: ${err.message}. Voice simulation and typing are available.`);
      setIsRecording(false);
    }
  };

  const handleSimulateVoice = () => {
    const sampleHearingVoices = [
      "Good morning, please take a seat.",
      "Can you tell me what symptoms you are experiencing?",
      "The pharmacy is on the ground floor to your left.",
      "Is there anyone accompanying you today?",
    ];
    const picked = sampleHearingVoices[Math.floor(Math.random() * sampleHearingVoices.length)];
    setInputText(picked);
    setActiveSender('hearing_user');
    setSpeechError(null);
  };

  const handleSendMessage = async (senderOverride?: 'deaf_user' | 'hearing_user') => {
    if (!inputText.trim()) return;

    const currentText = inputText.trim();
    const sender = senderOverride || activeSender;
    setInputText('');

    // Translate sentence via backend grammar and translation engine
    let translated: string | undefined = undefined;
    if (selectedLanguage === 'hi') {
      try {
        const res = await api.grammarCorrect(currentText.split(' '), selectedLanguage);
        translated = res.translated_sentence;
      } catch (err) {
        console.warn('Backend translation engine error, checking client fallback:', err);
      }

      if (!translated) {
        translated = getClientHindiFallback(currentText);
      }
    }

    const newMsg: ChatMessage = {
      id: String(Date.now()),
      sender,
      text: currentText,
      translatedText: translated,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, newMsg]);

    // If sent by deaf user, automatically speak text for the hearing user
    if (sender === 'deaf_user') {
      speakText(currentText, 'en');
    }
  };

  const handleCopyMessage = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 1500);
  };

  const handleDownloadTranscript = () => {
    const fullTranscript = messages
      .map(
        (m) =>
          `[${m.timestamp}] ${m.sender === 'deaf_user' ? 'Deaf User' : 'Hearing Interlocutor'}: ${m.text}${
            m.translatedText ? ` (Translation: ${m.translatedText})` : ''
          }`
      )
      .join('\n');

    const blob = new Blob([fullTranscript], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `signbridge-conversation-${new Date().toISOString().slice(0, 10)}.txt`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const handleSaveToAccount = async () => {
    const fullTranscript = messages
      .map((m) => `${m.sender === 'deaf_user' ? 'Deaf User' : 'Hearing Person'}: ${m.text}`)
      .join('\n');

    try {
      await api.saveTranscript({
        title: `Conversation (${new Date().toLocaleDateString()})`,
        content: fullTranscript,
        mode: 'conversation',
        language: selectedLanguage,
      });
      setSaveStatus('Conversation saved to your account!');
      setTimeout(() => setSaveStatus(null), 2500);
    } catch (err: any) {
      setSaveStatus(`Failed to save: ${err.message}`);
      setTimeout(() => setSaveStatus(null), 3000);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      {/* Header Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 bg-slate-900 border border-slate-800 rounded-3xl p-6">
        <div>
          <div className="flex items-center space-x-2">
            <span className="text-xs font-bold text-teal-400 uppercase tracking-wider">Bidirectional Hub</span>
            <span className="text-xs px-2 py-0.5 rounded-full bg-slate-800 text-slate-400 font-mono">
              Speech-to-Text & Text-to-Speech
            </span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white mt-1">
            Communication Workspace
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Facilitates smooth, dignified dialogue between deaf signers/typists and hearing speakers.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleDownloadTranscript}
            className="px-3.5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold flex items-center space-x-1.5 transition-colors border border-slate-700"
            title="Download Transcript as TXT file"
          >
            <Download className="w-4 h-4 text-teal-400" />
            <span>Export TXT</span>
          </button>
          <button
            onClick={handleSaveToAccount}
            className="px-3.5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold flex items-center space-x-1.5 transition-colors border border-slate-700"
            title="Save Transcript to Database"
          >
            <Save className="w-4 h-4 text-teal-400" />
            <span>Save</span>
          </button>
          <button
            onClick={() => setMessages([])}
            className="p-2 rounded-xl bg-slate-800 hover:bg-rose-900/40 text-slate-400 hover:text-rose-300 transition-colors border border-slate-700"
            title="Clear Chat Log"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {saveStatus && (
        <div className="p-3 rounded-xl bg-teal-500/10 border border-teal-500/30 text-teal-300 text-xs flex items-center space-x-2">
          <Check className="w-4 h-4" />
          <span>{saveStatus}</span>
        </div>
      )}

      {speechError && (
        <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs flex items-center space-x-2">
          <span>{speechError}</span>
        </div>
      )}

      {/* Main Conversation Canvas */}
      <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-xl flex flex-col h-[520px]">
        {/* Messages List Area */}
        <div className="flex-1 overflow-y-auto space-y-4 pr-2">
          {messages.map((m) => {
            const isDeaf = m.sender === 'deaf_user';
            return (
              <div
                key={m.id}
                className={`flex flex-col ${isDeaf ? 'items-end' : 'items-start'}`}
              >
                <div className="flex items-center space-x-2 text-[11px] text-slate-400 mb-1">
                  <span className="font-semibold text-slate-300">
                    {isDeaf ? 'Deaf User (Signed / Typed)' : 'Hearing Interlocutor (Voice)'}
                  </span>
                  <span>•</span>
                  <span>{m.timestamp}</span>
                </div>

                <div
                  className={`max-w-xl p-4 rounded-2xl relative group ${
                    isDeaf
                      ? 'bg-teal-950/40 border border-teal-500/30 text-teal-100 rounded-tr-none'
                      : 'bg-slate-950 border border-slate-800 text-slate-100 rounded-tl-none'
                  }`}
                >
                  <p className="text-sm leading-relaxed">{m.text}</p>
                  {m.translatedText && (
                    <div className="mt-2 pt-2 border-t border-slate-800/80 text-xs text-amber-300 flex items-center gap-1.5 font-medium">
                      <Languages className="w-3.5 h-3.5 flex-shrink-0" />
                      <span>{m.translatedText}</span>
                    </div>
                  )}

                  {/* Message hover actions: Speak and Copy */}
                  <div className="mt-2 pt-1 flex items-center justify-end space-x-1.5">
                    <button
                      onClick={() => speakText(m.text, 'en')}
                      className="p-1 rounded text-slate-400 hover:text-white"
                      title="Read Aloud in English"
                    >
                      <Volume2 className="w-3.5 h-3.5" />
                    </button>
                    {m.translatedText && (
                      <button
                        onClick={() => speakText(m.translatedText!, 'hi')}
                        className="p-1 rounded text-amber-400 hover:text-amber-300"
                        title="Read Aloud in Hindi"
                      >
                        <Volume2 className="w-3.5 h-3.5" />
                      </button>
                    )}
                    <button
                      onClick={() => handleCopyMessage(m.id, m.text)}
                      className="p-1 rounded text-slate-400 hover:text-white"
                      title="Copy Message Text"
                    >
                      {copiedId === m.id ? (
                        <Check className="w-3.5 h-3.5 text-emerald-400" />
                      ) : (
                        <Copy className="w-3.5 h-3.5" />
                      )}
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Input Composer Section */}
        <div className="pt-4 border-t border-slate-800 space-y-3">
          {/* Quick Phrases Bar */}
          <div className="flex items-center gap-1.5 overflow-x-auto pb-1 text-xs">
            <span className="text-[11px] text-slate-500 whitespace-nowrap">Quick Test Phrases:</span>
            {QUICK_PHRASES.map((qp, idx) => (
              <button
                key={idx}
                onClick={() => {
                  setInputText(qp.text);
                  setActiveSender(qp.sender);
                }}
                className="px-2.5 py-1 rounded-lg bg-slate-950 hover:bg-slate-800 text-slate-300 border border-slate-800 whitespace-nowrap text-[11px] transition-colors"
              >
                {qp.sender === 'deaf_user' ? '🤟 ' : '🗣️ '}{qp.text}
              </button>
            ))}
          </div>

          {/* Sender Role Toggle & Language Selection */}
          <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
            <div className="flex items-center gap-1 bg-slate-950 border border-slate-800 p-1 rounded-xl">
              <button
                type="button"
                onClick={() => setActiveSender('deaf_user')}
                className={`px-3 py-1 rounded-lg font-medium transition-all ${
                  activeSender === 'deaf_user'
                    ? 'bg-teal-500 text-slate-950 font-bold shadow'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                🤟 Signing/Typing (Deaf)
              </button>
              <button
                type="button"
                onClick={() => setActiveSender('hearing_user')}
                className={`px-3 py-1 rounded-lg font-medium transition-all ${
                  activeSender === 'hearing_user'
                    ? 'bg-amber-500 text-slate-950 font-bold shadow'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                🗣️ Speaking (Hearing)
              </button>
            </div>

            <div className="flex items-center space-x-2 text-slate-400">
              <Languages className="w-4 h-4 text-teal-400" />
              <span>Translation:</span>
              <select
                value={selectedLanguage}
                onChange={(e) => setSelectedLanguage(e.target.value)}
                className="bg-slate-950 border border-slate-800 rounded-lg px-2 py-1 text-white text-xs focus:outline-none"
              >
                <option value="hi">Hindi (Devanagari)</option>
                <option value="en">English (Original)</option>
              </select>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleToggleMic}
              className={`p-3 rounded-2xl border transition-colors ${
                isRecording
                  ? 'bg-rose-500 text-white border-rose-400 shadow-lg shadow-rose-500/25 animate-pulse'
                  : 'bg-slate-800 hover:bg-slate-700 text-slate-300 border-slate-700'
              }`}
              title={isRecording ? 'Stop Microphone' : 'Start Speech-to-Text via Microphone'}
            >
              {isRecording ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5 text-teal-400" />}
            </button>

            <button
              onClick={handleSimulateVoice}
              className="p-3 rounded-2xl bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 text-xs font-semibold whitespace-nowrap"
              title="Simulate speech from hearing interlocutor without microphone"
            >
              Simulate Voice
            </button>

            <input
              type="text"
              placeholder={
                activeSender === 'deaf_user'
                  ? 'Type as deaf user (will be read aloud to hearing user)...'
                  : 'Type or speak as hearing person (will be translated for deaf user)...'
              }
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  handleSendMessage();
                }
              }}
              className="flex-1 px-4 py-3 rounded-2xl bg-slate-950 border border-slate-800 text-white placeholder-slate-500 text-sm focus:outline-none focus:ring-2 focus:ring-teal-500"
            />

            <button
              onClick={() => handleSendMessage()}
              disabled={!inputText.trim()}
              className={`px-4 py-3 rounded-2xl font-bold text-xs flex items-center space-x-1 shadow-md transition-all ${
                activeSender === 'deaf_user'
                  ? 'bg-teal-500 hover:bg-teal-400 text-slate-950 shadow-teal-500/20'
                  : 'bg-amber-500 hover:bg-amber-400 text-slate-950 shadow-amber-500/20'
              } disabled:opacity-40`}
              title="Send message"
            >
              <Send className="w-4 h-4" />
              <span>{activeSender === 'deaf_user' ? 'Send & Speak' : 'Send Voice Reply'}</span>
            </button>
          </div>
        </div>
      </div>
    </div>

  );
};
