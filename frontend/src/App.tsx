import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { AuthProvider } from './contexts/AuthContext';
import ChatLayout from './components/ChatLayout';
import CompletionInterface from './components/CompletionInterface';
import StatusPanel from './components/StatusPanel';
import Navigation from './components/Navigation';
import AuthModal from './components/AuthModal';
import UserProfile from './components/UserProfile';
import './App.css';

interface AppState {
  lmStudioStatus: 'connected' | 'disconnected' | 'checking';
  modelStatus: 'loaded' | 'loading' | 'error';
  lastStatusCheck: number;
}

function App() {
  const [appState, setAppState] = useState<AppState>({
    lmStudioStatus: 'checking',
    modelStatus: 'loading',
    lastStatusCheck: 0
  });

  const [showAuthModal, setShowAuthModal] = useState(false);
  const [showUserProfile, setShowUserProfile] = useState(false);

  useEffect(() => {
    // Check initial status
    checkStatus();
    
    // Set up periodic status checks
    const interval = setInterval(checkStatus, 120000); // Check every 2 minutes instead of 30 seconds
    
    return () => clearInterval(interval);
  }, []);

  const checkStatus = async () => {
    const now = Date.now();
    
    // Avoid checking too frequently (throttle to at least 10 seconds between manual refreshes)
    if (now - appState.lastStatusCheck < 10000) {
      return;
    }

    try {
      setAppState(prev => ({ ...prev, lmStudioStatus: 'checking' }));
      
      // Check both LM Studio and model status from unified backend (port 8000)
      const lmResponse = await fetch('http://localhost:8000/api/lm-studio/status');
      const lmData = await lmResponse.json();
      
      const modelResponse = await fetch('http://localhost:8000/api/status');
      const modelData = await modelResponse.json();
      
      setAppState({
        lmStudioStatus: lmData.connected ? 'connected' : 'disconnected',
        modelStatus: modelData.loaded ? 'loaded' : 'error',
        lastStatusCheck: now
      });
    } catch (error) {
      setAppState({
        lmStudioStatus: 'disconnected',
        modelStatus: 'error',
        lastStatusCheck: now
      });
    }
  };

  return (
    <AuthProvider>
      <Router>
        <div className="app-container">
          <Routes>
            <Route path="/" element={
              <ChatLayout 
                lmStudioStatus={appState.lmStudioStatus}
                modelStatus={appState.modelStatus}
                onRefresh={checkStatus}
                onShowAuth={() => setShowAuthModal(true)}
                onShowProfile={() => setShowUserProfile(true)}
              />
            } />
            <Route path="/chat" element={
              <ChatLayout 
                lmStudioStatus={appState.lmStudioStatus}
                modelStatus={appState.modelStatus}
                onRefresh={checkStatus}
                onShowAuth={() => setShowAuthModal(true)}
                onShowProfile={() => setShowUserProfile(true)}
              />
            } />
            <Route path="/completion" element={
              <div className="main-content">
                <header className="header">
                  <div className="logo-container">
                    <div className="logo-circle">A</div>
                    <div className="logo-text">ATOM-GPT</div>
                  </div>
                  
                  <div className="header-controls">
                    <Navigation />
                    <StatusPanel 
                      lmStudioStatus={appState.lmStudioStatus}
                      modelStatus={appState.modelStatus}
                      onRefresh={checkStatus}
                      onShowAuth={() => setShowAuthModal(true)}
                      onShowProfile={() => setShowUserProfile(true)}
                    />
                  </div>
                </header>
                <CompletionInterface />
              </div>
            } />
          </Routes>
          
          {/* Auth Modal */}
          {showAuthModal && (
            <AuthModal 
              isOpen={showAuthModal}
              onClose={() => setShowAuthModal(false)}
            />
          )}
          
          {/* User Profile Modal */}
          {showUserProfile && (
            <UserProfile 
              isOpen={showUserProfile}
              onClose={() => setShowUserProfile(false)}
            />
          )}
        </div>
      </Router>
    </AuthProvider>
  );
}

export default App;
