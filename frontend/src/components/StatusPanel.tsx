import React from 'react';
import { useAuth } from '../contexts/AuthContext';

interface StatusPanelProps {
  lmStudioStatus: 'connected' | 'disconnected' | 'checking';
  modelStatus: 'loaded' | 'loading' | 'error';
  onRefresh: () => void;
  onShowAuth?: () => void;
  onShowProfile?: () => void;
}

const StatusPanel: React.FC<StatusPanelProps> = ({
  lmStudioStatus,
  modelStatus,
  onRefresh,
  onShowAuth,
  onShowProfile
}) => {
  const { user, isAuthenticated, logout } = useAuth();
  const getStatusClass = (status: string) => {
    switch (status) {
      case 'connected':
      case 'loaded':
        return 'connected';
      case 'disconnected':
      case 'error':
        return 'disconnected';
      case 'checking':
      case 'loading':
        return 'checking';
      default:
        return 'disconnected';
    }
  };

  return (
    <div className="status-panel">
      <div className="status-item">
        <div className={`status-indicator ${getStatusClass(lmStudioStatus)}`}></div>
        <span>LM Studio: {lmStudioStatus}</span>
      </div>
      
      <div className="status-item">
        <div className={`status-indicator ${getStatusClass(modelStatus)}`}></div>
        <span>Model: {modelStatus}</span>
      </div>

      <button
        onClick={onRefresh}
        className="refresh-button"
        title="Refresh Status"
      >
        ↻
      </button>

      {/* Authentication UI */}
      <div className="auth-section">
        {isAuthenticated ? (
          <div className="user-section">
            <button
              onClick={onShowProfile}
              className="user-button"
              title="User Profile"
            >
              <div className="user-avatar">{user?.username?.[0]?.toUpperCase()}</div>
              <span className="username">{user?.username}</span>
            </button>
            <button
              onClick={logout}
              className="logout-button"
              title="Logout"
            >
              ⏻
            </button>
          </div>
        ) : (
          <button
            onClick={onShowAuth}
            className="login-button"
            title="Login / Register"
          >
            👤 Login
          </button>
        )}
      </div>
    </div>
  );
};

export default StatusPanel;
