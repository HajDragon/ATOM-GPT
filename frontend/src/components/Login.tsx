import React, { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';

interface LoginProps {
  onSwitchToRegister: () => void;
  onClose?: () => void;
}

const Login: React.FC<LoginProps> = ({ onSwitchToRegister, onClose }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const { login, error, clearError } = useAuth();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (isSubmitting) return;

    setIsSubmitting(true);
    clearError();

    try {
      await login(email, password);
      onClose?.();
    } catch (err) {
      // Error is handled by AuthContext
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleDemoLogin = async () => {
    setEmail('admin@atomgpt.local');
    setPassword('admin123');
    
    try {
      await login('admin@atomgpt.local', 'admin123');
      onClose?.();
    } catch (err) {
      // Error is handled by AuthContext
    }
  };

  return (
    <div>
      <div className="modal-header">
        <h1 className="modal-title">🎸 Sign In to ATOM-GPT</h1>
      </div>

      <form onSubmit={handleSubmit} className="auth-form">
        {error && (
          <div className="error-message">
            ⚠️ {error}
          </div>
        )}

        <div className="form-group">
          <label htmlFor="email" className="form-label">Email</label>
          <input
            type="email"
            id="email"
            className="form-input"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="your.email@domain.com"
            required
            disabled={isSubmitting}
          />
        </div>

        <div className="form-group">
          <label htmlFor="password" className="form-label">Password</label>
          <input
            type="password"
            id="password"
            className="form-input"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Enter your password"
            required
            disabled={isSubmitting}
          />
        </div>

        <button
          type="submit"
          className="form-button"
          disabled={isSubmitting}
        >
          {isSubmitting ? (
            <>
              <span className="loading-spinner"></span>
              Signing In...
            </>
          ) : (
            '🚀 Sign In'
          )}
        </button>

        <button
          type="button"
          className="form-button secondary"
          onClick={handleDemoLogin}
          disabled={isSubmitting}
        >
          🎭 Try Demo Account
        </button>
      </form>

      <div className="auth-toggle">
        <p>
          Don't have an account?{' '}
          <button
            type="button"
            onClick={onSwitchToRegister}
            disabled={isSubmitting}
          >
            Create one here
          </button>
        </p>
      </div>
    </div>
  );
};

export default Login;
