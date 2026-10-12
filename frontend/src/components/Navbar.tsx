import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import {
  Video,
  BookOpen,
  MessageSquare,
  Captions,
  AlertTriangle,
  BarChart3,
  Settings,
  LayoutDashboard,
  LogIn,
  LogOut,
  Menu,
  X,
  Volume2,
  Eye,
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { useAccessibility } from '../context/AccessibilityContext';

export const Navbar: React.FC = () => {
  const { user, logout } = useAuth();
  const { highContrast, toggleHighContrast, largeText, toggleLargeText } = useAccessibility();
  const location = useLocation();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navItems = [
    { label: 'Recognize', path: '/recognize', icon: Video },
    { label: 'Tutor', path: '/tutor', icon: BookOpen },
    { label: 'Chat', path: '/communication', icon: MessageSquare },
    { label: 'Captions', path: '/captions', icon: Captions },
    { label: 'Emergency', path: '/emergency', icon: AlertTriangle, highlight: true },
    { label: 'Analytics', path: '/analytics', icon: BarChart3 },
  ];

  const isActive = (path: string) => location.pathname === path;

  return (
    <header className="sticky top-0 z-50 bg-slate-900/90 backdrop-blur-md border-b border-slate-800 transition-colors">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Brand */}
          <Link to="/" className="flex items-center space-x-3 group focus:outline-none focus:ring-2 focus:ring-teal-400 rounded-lg p-1">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-teal-600 to-emerald-400 flex items-center justify-center shadow-lg shadow-teal-500/20 group-hover:scale-105 transition-transform">
              <span className="text-white font-extrabold text-lg">S</span>
            </div>
            <div>
              <span className="text-lg font-bold tracking-tight text-white flex items-center gap-1.5">
                SignBridge <span className="text-teal-400 text-xs font-semibold px-1.5 py-0.5 rounded bg-teal-950/80 border border-teal-800/80">ISL AI</span>
              </span>
            </div>
          </Link>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center space-x-1 lg:space-x-2">
            {navItems.map((item) => {
              const Icon = item.icon;
              const active = isActive(item.path);
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`flex items-center space-x-1.5 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                    active
                      ? 'bg-teal-500/15 text-teal-300 border border-teal-500/30'
                      : item.highlight
                      ? 'text-rose-400 hover:bg-rose-500/10 hover:text-rose-300'
                      : 'text-slate-300 hover:bg-slate-800/70 hover:text-white'
                  }`}
                >
                  <Icon className={`w-4 h-4 ${item.highlight && !active ? 'text-rose-400' : ''}`} />
                  <span>{item.label}</span>
                </Link>
              );
            })}
          </nav>

          {/* Right Utility & Auth Actions */}
          <div className="hidden sm:flex items-center space-x-2">
            {/* Accessibility Quick Toggles */}
            <div className="flex items-center bg-slate-800/80 rounded-lg p-1 border border-slate-700/80 text-xs">
              <button
                onClick={toggleHighContrast}
                title="Toggle High Contrast"
                className={`p-1.5 rounded transition-colors ${
                  highContrast ? 'bg-amber-400 text-slate-950 font-bold' : 'text-slate-400 hover:text-white'
                }`}
                aria-label="Toggle High Contrast"
              >
                <Eye className="w-4 h-4" />
              </button>
              <button
                onClick={toggleLargeText}
                title="Toggle Large Text"
                className={`px-2 py-1 rounded transition-colors font-semibold ${
                  largeText ? 'bg-teal-500 text-white' : 'text-slate-400 hover:text-white'
                }`}
                aria-label="Toggle Large Text"
              >
                A+
              </button>
            </div>

            {user ? (
              <div className="flex items-center space-x-2">
                <Link
                  to="/dashboard"
                  className={`flex items-center space-x-1.5 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                    isActive('/dashboard') ? 'bg-teal-500/20 text-teal-300' : 'text-slate-300 hover:bg-slate-800'
                  }`}
                >
                  <LayoutDashboard className="w-4 h-4" />
                  <span className="hidden lg:inline">Dashboard</span>
                </Link>
                <Link
                  to="/settings"
                  className="p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg"
                  title="Settings"
                >
                  <Settings className="w-4 h-4" />
                </Link>
                <button
                  onClick={logout}
                  className="flex items-center space-x-1 text-slate-400 hover:text-rose-400 p-2 rounded-lg"
                  title="Logout"
                >
                  <LogOut className="w-4 h-4" />
                </button>
              </div>
            ) : (
              <div className="flex items-center space-x-2">
                <Link
                  to="/login"
                  className="flex items-center space-x-1 px-3 py-1.5 rounded-lg text-sm font-medium text-slate-300 hover:bg-slate-800 hover:text-white"
                >
                  <LogIn className="w-4 h-4" />
                  <span>Sign In</span>
                </Link>
                <Link
                  to="/register"
                  className="px-3.5 py-1.5 rounded-lg text-sm font-medium bg-teal-500 hover:bg-teal-400 text-slate-950 font-semibold shadow-md shadow-teal-500/20 transition-colors"
                >
                  Get Started
                </Link>
              </div>
            )}
          </div>

          {/* Mobile menu toggle */}
          <div className="flex md:hidden items-center space-x-2">
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800"
              aria-label="Open main menu"
            >
              {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Drawer */}
      {mobileMenuOpen && (
        <div className="md:hidden border-t border-slate-800 bg-slate-900/95 px-4 pt-3 pb-6 space-y-2">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <Link
                key={item.path}
                to={item.path}
                onClick={() => setMobileMenuOpen(false)}
                className={`flex items-center space-x-3 px-3 py-2.5 rounded-lg text-base font-medium ${
                  isActive(item.path)
                    ? 'bg-teal-500/20 text-teal-300 font-semibold'
                    : 'text-slate-300 hover:bg-slate-800'
                }`}
              >
                <Icon className="w-5 h-5" />
                <span>{item.label}</span>
              </Link>
            );
          })}
          <div className="pt-4 border-t border-slate-800 flex flex-col space-y-2">
            {user ? (
              <>
                <Link
                  to="/dashboard"
                  onClick={() => setMobileMenuOpen(false)}
                  className="flex items-center space-x-3 px-3 py-2 text-slate-300 hover:bg-slate-800 rounded-lg"
                >
                  <LayoutDashboard className="w-5 h-5" />
                  <span>User Dashboard</span>
                </Link>
                <Link
                  to="/settings"
                  onClick={() => setMobileMenuOpen(false)}
                  className="flex items-center space-x-3 px-3 py-2 text-slate-300 hover:bg-slate-800 rounded-lg"
                >
                  <Settings className="w-5 h-5" />
                  <span>Accessibility & Profile</span>
                </Link>
                <button
                  onClick={() => {
                    logout();
                    setMobileMenuOpen(false);
                  }}
                  className="flex items-center space-x-3 px-3 py-2 text-rose-400 hover:bg-slate-800 rounded-lg text-left"
                >
                  <LogOut className="w-5 h-5" />
                  <span>Logout</span>
                </button>
              </>
            ) : (
              <div className="grid grid-cols-2 gap-2 pt-2">
                <Link
                  to="/login"
                  onClick={() => setMobileMenuOpen(false)}
                  className="px-4 py-2.5 rounded-lg text-center text-sm font-medium bg-slate-800 text-slate-200"
                >
                  Sign In
                </Link>
                <Link
                  to="/register"
                  onClick={() => setMobileMenuOpen(false)}
                  className="px-4 py-2.5 rounded-lg text-center text-sm font-medium bg-teal-500 text-slate-950 font-semibold"
                >
                  Get Started
                </Link>
              </div>
            )}
          </div>
        </div>
      )}
    </header>
  );
};
