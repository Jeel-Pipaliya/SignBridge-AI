import React, { useState, useEffect } from 'react';
import {
  BarChart3,
  TrendingUp,
  Cpu,
  Download,
  Flame,
  Award,
  CheckCircle2,
  Layers,
  Sparkles,
  Info,
} from 'lucide-react';
import { api } from '../api/client';
import { AnalyticsOverview } from '../api/types';
import { StatCard } from '../components/StatCard';

export const AnalyticsPage: React.FC = () => {
  const [analytics, setAnalytics] = useState<AnalyticsOverview | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    api.getAnalyticsOverview()
      .then((data) => setAnalytics(data))
      .catch((err) => console.warn('Could not load analytics:', err))
      .finally(() => setLoading(false));
  }, []);

  const handleExportData = () => {
    if (!analytics) return;
    const blob = new Blob([JSON.stringify(analytics, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `signbridge-analytics-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Header Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 bg-slate-900 border border-slate-800 rounded-3xl p-6 sm:p-8">
        <div>
          <div className="flex items-center space-x-2">
            <span className="text-xs font-bold text-teal-400 uppercase tracking-wider">Evaluation Telemetry</span>
            <span className="text-xs px-2 py-0.5 rounded-full bg-slate-800 text-slate-400 font-mono">
              Offline vs. Live Benchmark
            </span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white mt-1">
            Analytics & Recognition Performance
          </h1>
          <p className="text-xs sm:text-sm text-slate-400 mt-1 max-w-xl">
            Detailed evaluation comparing scientifically validated offline test set benchmarks with live student practice telemetry.
          </p>
        </div>

        <button
          onClick={handleExportData}
          disabled={!analytics}
          className="px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold flex items-center space-x-2 border border-slate-700 transition-colors"
        >
          <Download className="w-4 h-4 text-teal-400" />
          <span>Export Metrics JSON</span>
        </button>
      </div>

      {/* Primary Metrics Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Held-Out Test Accuracy"
          value="66.07%"
          subtitle="Evaluated on 32 ISL classes"
          icon={Cpu}
          badge="ML Benchmark"
          badgeColor="emerald"
        />
        <StatCard
          title="Macro F1 Score"
          value="0.6085"
          subtitle="Harmonic mean of precision & recall"
          icon={Award}
          badge="Scientific Metric"
          badgeColor="teal"
        />
        <StatCard
          title="Dynamic Bi-LSTM Acc"
          value="100.0%"
          subtitle="Sequence gestures (HELLO, J, Z)"
          icon={TrendingUp}
          badge="Temporal Model"
          badgeColor="blue"
        />
        <StatCard
          title="Interactive User Accuracy"
          value={analytics ? `${(analytics.average_practice_accuracy * 100).toFixed(1)}%` : '0.0%'}
          subtitle={`Across ${analytics?.total_practice_sessions || 0} user sessions`}
          icon={BarChart3}
          badge="Live Practice"
          badgeColor="amber"
        />
      </div>

      {/* Benchmark Matrix Card */}
      <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 sm:p-8 space-y-6 shadow-xl">
        <div className="flex items-center justify-between border-b border-slate-800 pb-4">
          <div className="flex items-center space-x-3">
            <div className="p-2.5 rounded-xl bg-teal-500/10 text-teal-400">
              <Layers className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white">Model Architecture & Benchmark Comparison</h2>
              <p className="text-xs text-slate-400">
                Ground truth measurements conducted on invariant 63D normalized hand landmarks.
              </p>
            </div>
          </div>
          <span className="text-xs font-mono text-emerald-400 bg-emerald-950/40 px-3 py-1 rounded-full border border-emerald-800/60">
            Validated Offline Checkpoints
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="p-5 rounded-2xl bg-slate-950 border border-slate-800 space-y-3">
            <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">Champion Static Classifier</span>
            <p className="text-xl font-extrabold text-white">MLP Neural Network</p>
            <div className="space-y-1.5 text-xs text-slate-400 pt-2 border-t border-slate-800">
              <div className="flex justify-between"><span>Classes:</span><span className="text-white font-mono">32 ISL Signs</span></div>
              <div className="flex justify-between"><span>Training Accuracy:</span><span className="text-emerald-400 font-mono">73.08%</span></div>
              <div className="flex justify-between"><span>Validation Accuracy:</span><span className="text-emerald-400 font-mono">69.05%</span></div>
              <div className="flex justify-between"><span>Held-Out Test Accuracy:</span><span className="text-emerald-400 font-mono font-bold">66.07%</span></div>
              <div className="flex justify-between"><span>Test Macro F1:</span><span className="text-teal-400 font-mono">0.6085</span></div>
            </div>
          </div>

          <div className="p-5 rounded-2xl bg-slate-950 border border-slate-800 space-y-3">
            <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">Dynamic Sequence Model</span>
            <p className="text-xl font-extrabold text-white">Bi-LSTM Classifier</p>
            <div className="space-y-1.5 text-xs text-slate-400 pt-2 border-t border-slate-800">
              <div className="flex justify-between"><span>Temporal Classes:</span><span className="text-white font-mono">HELLO, J, Z</span></div>
              <div className="flex justify-between"><span>Sequence Window:</span><span className="text-white font-mono">30 Frames (1.0s)</span></div>
              <div className="flex justify-between"><span>Epochs Trained:</span><span className="text-white font-mono">30 Epochs</span></div>
              <div className="flex justify-between"><span>Validation Accuracy:</span><span className="text-emerald-400 font-mono">100.0%</span></div>
              <div className="flex justify-between"><span>Test Accuracy:</span><span className="text-emerald-400 font-mono font-bold">100.0%</span></div>
            </div>
          </div>

          <div className="p-5 rounded-2xl bg-slate-950 border border-slate-800 space-y-3">
            <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">Feature Pipeline Invariance</span>
            <p className="text-xl font-extrabold text-white">63-D Coordinate Vector</p>
            <div className="space-y-1.5 text-xs text-slate-400 pt-2 border-t border-slate-800">
              <div className="flex justify-between"><span>Translation Invariance:</span><span className="text-emerald-400 font-mono">PASSED (&lt;1e-6)</span></div>
              <div className="flex justify-between"><span>Scale Invariance:</span><span className="text-emerald-400 font-mono">PASSED (&lt;1e-6)</span></div>
              <div className="flex justify-between"><span>Combined Invariance:</span><span className="text-emerald-400 font-mono">PASSED (&lt;1e-6)</span></div>
              <div className="flex justify-between"><span>Inference Latency:</span><span className="text-teal-400 font-mono font-bold">&lt; 15 ms</span></div>
              <div className="flex justify-between"><span>Automated Tests:</span><span className="text-emerald-400 font-mono">96 / 96 Passed</span></div>
            </div>
          </div>
        </div>
      </div>

      {/* Most Practiced Vocabulary Frequency Chart */}
      <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 sm:p-8 space-y-4">
        <h2 className="text-base font-bold text-white uppercase tracking-wider">
          Student Practice Frequency Breakdown
        </h2>

        {analytics?.most_practiced_signs && analytics.most_practiced_signs.length > 0 ? (
          <div className="space-y-3">
            {analytics.most_practiced_signs.map((item) => (
              <div key={item.sign} className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="font-bold text-white">Sign "{item.sign}" ({item.category})</span>
                  <span className="text-slate-400 font-mono">
                    {item.practice_count} sessions • {(item.success_rate * 100).toFixed(0)}% accurate
                  </span>
                </div>
                <div className="w-full h-2 rounded-full bg-slate-950 overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-teal-500 to-emerald-400 rounded-full"
                    style={{ width: `${Math.min(item.success_rate * 100, 100)}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-xs text-slate-500 py-6 text-center">
            No sign breakdown data recorded yet. Practice signs in the Tutor to populate this chart!
          </p>
        )}
      </div>
    </div>
  );
};
