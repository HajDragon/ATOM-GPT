import React, { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';

interface UserProfileProps {
  isOpen: boolean;
  onClose: () => void;
}

const UserProfile: React.FC<UserProfileProps> = ({ isOpen, onClose }) => {
  const { user, stats, logout, refreshUserData } = useAuth();
  const [isRefreshing, setIsRefreshing] = useState(false);

  if (!isOpen || !user) return null;

  const handleRefresh = async () => {
    setIsRefreshing(true);
    try {
      await refreshUserData();
    } catch (err) {
      console.error('Failed to refresh user data:', err);
    } finally {
      setIsRefreshing(false);
    }
  };

  const handleOverlayClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  return (
    <div className="modal-overlay" onClick={handleOverlayClick}>
      <div className="modal-content">
        <div className="modal-header">
          <h1 className="modal-title">👤 User Profile</h1>
          <button className="modal-close" onClick={onClose}>×</button>
        </div>

        <div className="profile-content">
          <div className="profile-section">
            <h3>Account Information</h3>
            <div className="profile-field">
              <label>Name</label>
              <span>{user.first_name && user.last_name 
                ? `${user.first_name} ${user.last_name}` 
                : user.username}
              </span>
            </div>
            <div className="profile-field">
              <label>Email</label>
              <span>{user.email}</span>
            </div>
            <div className="profile-field">
              <label>Username</label>
              <span>@{user.username}</span>
            </div>
            <div className="profile-field">
              <label>Account Type</label>
              <span>{user.is_admin ? '🔑 Admin' : '👤 User'}</span>
            </div>
            <div className="profile-field">
              <label>Member Since</label>
              <span>{new Date(user.created_at).toLocaleDateString()}</span>
            </div>
          </div>

          {stats && (
            <div className="profile-section">
              <h3>📊 Usage Statistics</h3>
              <div className="stats-grid">
                <div className="stat-item">
                  <span className="stat-value">{stats.total_conversations}</span>
                  <span className="stat-label">Conversations</span>
                </div>
                
                <div className="stat-item">
                  <span className="stat-value">{stats.total_messages}</span>
                  <span className="stat-label">Messages</span>
                </div>
                
                <div className="stat-item">
                  <span className="stat-value">{stats.total_tokens?.toLocaleString() || 0}</span>
                  <span className="stat-label">Tokens Used</span>
                </div>
                
                <div className="stat-item">
                  <span className="stat-value">{Math.round(stats.avg_response_time || 0)}ms</span>
                  <span className="stat-label">Avg Response</span>
                </div>
                
                <div className="stat-item">
                  <span className="stat-value">{stats.enhanced_requests}</span>
                  <span className="stat-label">Enhanced</span>
                </div>
                
                <div className="stat-item">
                  <span className="stat-value">{stats.total_requests}</span>
                  <span className="stat-label">API Calls</span>
                </div>
              </div>
            </div>
          )}

          <div className="profile-actions">
            <button
              className="form-button secondary"
              onClick={handleRefresh}
              disabled={isRefreshing}
            >
              {isRefreshing ? (
                <>
                  <span className="loading-spinner"></span>
                  Refreshing...
                </>
              ) : (
                '📊 Refresh Stats'
              )}
            </button>
            
            <button className="form-button" onClick={logout}>
              🚪 Sign Out
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default UserProfile;
