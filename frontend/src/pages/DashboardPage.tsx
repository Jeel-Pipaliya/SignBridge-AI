import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Video,
  BookOpen,
  MessageSquare,
  AlertTriangle,
  Award,
  Flame,
  BarChart3,
  CheckCircle2,
  Clock,
  ArrowRight,
  TrendingUp,
  Cpu,
  RefreshCw,
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { api } from '../api/client';
import { AnalyticsOverview, PracticeSummary } from '../api/types';
import { StatCard } from '../components/StatCard';

export const DashboardPage: React.FC = () => {
  const { user } = useAuth();
  const [analytics, setAnalytics] = useState<AnalyticsOverview | null>(null);
  const [summary, setSummary] = useState<PracticeSummary | null>(null);
  const [loading, setLoading] = useState(true);

  const loadData = async () => {
    setLoading(true);
    try {
      const [ana, sum] = await Promise.all([
        api.getAnalyticsOverview(),
        api.getPracticeSummary(),
      ]);
      setAnalytics(ana);
      setSummary(sum);
    } catch (err) {
      console.warn('Error loading dashboard stats:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Welcome & Status Bar */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 bg-slate-900 border border-slate-800 rounded-3xl p-6 sm:p-8 relative overflow-hidden">
        <div className="space-y-1 z-10">
          <div className="flex items-center space-x-2">
            <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-teal-500/10 text-teal-400 border border-teal-500/20">
              Active Student Session
            </span>
            <span className="text-xs text-slate-400 font-mono">ID: #{user?.id || 1}</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white">
            Welcome back, {user?.full_name || user?.email.split('@')[0]}!
          </h1>
          <p className="text-xs sm:text-sm text-slate-400 max-w-xl">
            Track your Indian Sign Language communication practice, review evaluated accuracy metrics, and launch recognition tools.
          </p>
        </div>

        <div className="flex items-center gap-3 z-10">
          <button
            onClick={loadData}
            className="p-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
            title="Refresh Telemetry"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
          <Link
            to="/recognize"
            className="px-5 py-2.5 rounded-xl bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold text-xs sm:text-sm flex items-center space-x-2 shadow-lg shadow-teal-500/20 transition-transform hover:scale-102"
          >
            <Video className="w-4 h-4" />
            <span>Launch Recognition</span>
          </Link>
        </div>
      </div>

      {/* Primary Telemetry Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Practice Streak"
          value={summary ? `${summary.current_streak} Signs` : '0 Signs'}
          subtitle="Consecutive accurate gestures"
          icon={Flame}
          badge="Streak"
          badgeColor="amber"
        />
        <StatCard
          title="Overall Practice Accuracy"
          value={summary ? `${(summary.overall_accuracy * 100).toFixed(1)}%` : '0.0%'}
          subtitle={`${summary?.correct_count || 0} of ${summary?.total_sessions || 0} correct`}
          icon={TrendingUp}
          badge="Live User"
          badgeColor="teal"
        />
        <StatCard
          title="Offline Model Benchmark"
          value="66.07%"
          subtitle="Held-out 32-class test set accuracy"
          icon={Cpu}
          badge="Ground Truth"
          badgeColor="emerald"
        />
        <StatCard
          title="Recognitions Logged"
          value={analytics?.total_recognitions || 0}
          subtitle="Total vision inferences stored"
          icon={BarChart3}
          badge="Telemetry"
          badgeColor="blue"
        />
      </div>

      {/* Offline Benchmark vs Live Real-World Accuracy Callout */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div className="flex items-start space-x-3">
          <div className="p-2 rounded-xl bg-teal-500/10 text-teal-400 flex-shrink-0 mt-0.5">
            <CheckCircle2 className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-white">Scientific Evaluation Transparency Note</h3>
            <p className="text-xs text-slate-400 mt-0.5 leading-relaxed">
              {analytics?.offline_vs_live_note ||
                'Offline test-set accuracy (66.07%, F1: 0.6085) is scientifically computed on held-out test splits. Live user accuracy reflects camera conditions and human precision.'}
            </p>
          </div>
        </div>
        <Link
          to="/analytics"
          className="text-xs font-semibold text-teal-400 hover:text-teal-300 flex items-center space-x-1 whitespace-nowrap px-3 py-1.5 rounded-lg bg-teal-950/40 border border-teal-800/60"
        >
          <span>Examine Model Matrix</span>
          <ArrowRight className="w-3.5 h-3.5" />
        </Link>
      </div>

      {/* Two Column Section: Quick Access & Recent Practice Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Quick Feature Launchers */}
        <div className="space-y-4">
          <h2 className="text-sm font-bold text-white uppercase tracking-wider">Quick Actions</h2>
          <div className="space-y-3">
            <Link
              to="/recognize"
              className="flex items-center justify-between p-4 rounded-2xl bg-slate-900 border border-slate-800 hover:border-teal-500/40 transition-colors group"
            >
              <div className="flex items-center space-x-3">
                <div className="p-2.5 rounded-xl bg-teal-500/10 text-teal-400 group-hover:bg-teal-500 group-hover:text-slate-950 transition-colors">
                  <Video className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="text-sm font-bold text-white">Live Recognition</h3>
                  <p className="text-xs text-slate-400">Webcam stream to sentence builder & speech</p>
                </div>
              </div>
              <ArrowRight className="w-4 h-4 text-slate-500 group-hover:text-teal-400 transition-colors" />
            </Link>

            <Link
              to="/tutor"
              className="flex items-center justify-between p-4 rounded-2xl bg-slate-900 border border-slate-800 hover:border-teal-500/40 transition-colors group"
            >
              <div className="flex items-center space-x-3">
                <div className="p-2.5 rounded-xl bg-teal-500/10 text-teal-400 group-hover:bg-teal-500 group-hover:text-slate-950 transition-colors">
                  <BookOpen className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="text-sm font-bold text-white">Sign Learning Tutor</h3>
                  <p className="text-xs text-slate-400">Practice 32 ISL signs with real-time scoring</p>
                </div>
              </div>
              <ArrowRight className="w-4 h-4 text-slate-500 group-hover:text-teal-400 transition-colors" />
            </Link>

            <Link
              to="/communication"
              className="flex items-center justify-between p-4 rounded-2xl bg-slate-900 border border-slate-800 hover:border-teal-500/40 transition-colors group"
            >
              <div className="flex items-center space-x-3">
                <div className="p-2.5 rounded-xl bg-teal-500/10 text-teal-400 group-hover:bg-teal-500 group-hover:text-slate-950 transition-colors">
                  <MessageSquare className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="text-sm font-bold text-white">Two-Way Speech Hub</h3>
                  <p className="text-xs text-slate-400">Type, speak, translate between English & Hindi</p>
                </div>
              </div>
              <ArrowRight className="w-4 h-4 text-slate-500 group-hover:text-teal-400 transition-colors" />
            </Link>

            <Link
              to="/emergency"
              className="flex items-center justify-between p-4 rounded-2xl bg-rose-950/20 border border-rose-500/30 hover:border-rose-400 transition-colors group"
            >
              <div className="flex items-center space-x-3">
                <div className="p-2.5 rounded-xl bg-rose-500/10 text-rose-400 group-hover:bg-rose-500 group-hover:text-slate-950 transition-colors">
                  <AlertTriangle className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="text-sm font-bold text-white">Emergency Assist Mode</h3>
                  <p className="text-xs text-rose-300">Preset phrases with instant loud speech alarm</p>
                </div>
              </div>
              <ArrowRight className="w-4 h-4 text-rose-500 group-hover:text-rose-400 transition-colors" />
            </Link>
          </div>
        </div>

        {/* Recent Practice History Timeline */}
        <div className="lg:col-span-2 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-bold text-white uppercase tracking-wider">Recent Practice Evaluations</h2>
            <Link to="/tutor" className="text-xs font-semibold text-teal-400 hover:underline">
              Practice More Signs
            </Link>
          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-3xl p-5 shadow-sm divide-y divide-slate-800">
            {analytics?.recent_activity_timeline && analytics.recent_activity_timeline.length > 0 ? (
              analytics.recent_activity_timeline.map((act) => (
                <div key={act.id} className="py-3.5 first:pt-0 last:pb-0 flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div
                      className={`w-9 h-9 rounded-xl flex items-center justify-center font-bold text-sm ${
                        act.is_correct
                          ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30'
                          : 'bg-rose-500/15 text-rose-400 border border-rose-500/30'
                      }`}
                    >
                      {act.target}
                    </div>
                    <div>
                      <div className="flex items-center space-x-2">
                        <span className="text-sm font-semibold text-white">Sign "{act.target}"</span>
                        <span
                          className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${
                            act.is_correct
                              ? 'bg-emerald-500/10 text-emerald-300'
                              : 'bg-rose-500/10 text-rose-300'
                          }`}
                        >
                          {act.is_correct ? 'Accurate' : 'Needs Practice'}
                        </span>
                      </div>
                      <span className="text-xs text-slate-500 flex items-center gap-1 mt-0.5">
                        <Clock className="w-3 h-3 inline" />
                        {new Date(act.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} •{' '}
                        {new Date(act.timestamp).toLocaleDateString()}
                      </span>
                    </div>
                  </div>

                  <div className="text-right">
                    <span className="text-sm font-bold text-white">{(act.accuracy * 100).toFixed(0)}%</span>
                    <p className="text-[10px] text-slate-500">Similarity Score</p>
                  </div>
                </div>
              ))
            ) : (
              <div className="py-12 text-center text-slate-500 space-y-2">
                <BookOpen className="w-8 h-8 mx-auto text-slate-600" />
                <p className="text-sm">No practice sessions recorded yet.</p>
                <Link to="/tutor" className="text-xs text-teal-400 font-semibold hover:underline">
                  Launch the tutor to start learning!
                </Link>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
