import React, { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';

interface RegisterProps {
  onSwitchToLogin: () => void;
  onClose?: () => void;
}

const Register: React.FC<RegisterProps> = ({ onSwitchToLogin, onClose }) => {
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: '',
    confirmPassword: '',
    first_name: '',
    last_name: ''
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [validationErrors, setValidationErrors] = useState<Record<string, string>>({});
  const { register, error, clearError } = useAuth();

  const validateForm = () => {
    const errors: Record<string, string> = {};

    if (formData.username.length < 3) {
      errors.username = 'Username must be at least 3 characters';
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(formData.email)) {
      errors.email = 'Please enter a valid email address';
    }

    if (formData.password.length < 6) {
      errors.password = 'Password must be at least 6 characters';
    }

    if (!/[A-Za-z]/.test(formData.password)) {
      errors.password = 'Password must contain at least one letter';
    }

    if (!/\d/.test(formData.password)) {
      errors.password = 'Password must contain at least one number';
    }

    if (formData.password !== formData.confirmPassword) {
      errors.confirmPassword = 'Passwords do not match';
    }

    setValidationErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (isSubmitting) return;

    if (!validateForm()) return;

    setIsSubmitting(true);
    clearError();

    try {
      await register({
        username: formData.username,
        email: formData.email,
        password: formData.password,
        first_name: formData.first_name,
        last_name: formData.last_name
      });
      onClose?.();
    } catch (err) {
      // Error is handled by AuthContext
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    
    // Clear validation error when user starts typing
    if (validationErrors[field]) {
      setValidationErrors(prev => ({ ...prev, [field]: '' }));
    }
  };

  return (
    <div>
      <div className="modal-header">
        <h1 className="modal-title">🤘 Join ATOM-GPT</h1>
      </div>

      <form onSubmit={handleSubmit} className="auth-form">
        {error && (
          <div className="error-message">
            ⚠️ {error}
          </div>
        )}

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
          <div className="form-group">
            <label htmlFor="first_name" className="form-label">First Name</label>
            <input
              type="text"
              id="first_name"
              className="form-input"
              value={formData.first_name}
              onChange={(e) => handleInputChange('first_name', e.target.value)}
              placeholder="First name"
              disabled={isSubmitting}
            />
          </div>

          <div className="form-group">
            <label htmlFor="last_name" className="form-label">Last Name</label>
            <input
              type="text"
              id="last_name"
              className="form-input"
              value={formData.last_name}
              onChange={(e) => handleInputChange('last_name', e.target.value)}
              placeholder="Last name"
              disabled={isSubmitting}
            />
          </div>
        </div>

        <div className="form-group">
          <label htmlFor="username" className="form-label">Username *</label>
          <input
            type="text"
            id="username"
            className={`form-input ${validationErrors.username ? 'error' : ''}`}
            value={formData.username}
            onChange={(e) => handleInputChange('username', e.target.value)}
            placeholder="Choose a username"
            required
            disabled={isSubmitting}
          />
          {validationErrors.username && (
            <span style={{ color: '#fecaca', fontSize: '0.8rem', marginTop: '0.25rem', display: 'block' }}>
              {validationErrors.username}
            </span>
          )}
        </div>

        <div className="form-group">
          <label htmlFor="email" className="form-label">Email *</label>
          <input
            type="email"
            id="email"
            className={`form-input ${validationErrors.email ? 'error' : ''}`}
            value={formData.email}
            onChange={(e) => handleInputChange('email', e.target.value)}
            placeholder="your.email@domain.com"
            required
            disabled={isSubmitting}
          />
          {validationErrors.email && (
            <span style={{ color: '#fecaca', fontSize: '0.8rem', marginTop: '0.25rem', display: 'block' }}>
              {validationErrors.email}
            </span>
          )}
        </div>

        <div className="form-group">
          <label htmlFor="password" className="form-label">Password *</label>
          <input
            type="password"
            id="password"
            className={`form-input ${validationErrors.password ? 'error' : ''}`}
            value={formData.password}
            onChange={(e) => handleInputChange('password', e.target.value)}
            placeholder="Create a strong password"
            required
            disabled={isSubmitting}
          />
          {validationErrors.password && (
            <span style={{ color: '#fecaca', fontSize: '0.8rem', marginTop: '0.25rem', display: 'block' }}>
              {validationErrors.password}
            </span>
          )}
          <div style={{ color: '#888', fontSize: '0.8rem', marginTop: '0.25rem' }}>
            At least 6 characters with letters and numbers
          </div>
        </div>

        <div className="form-group">
          <label htmlFor="confirmPassword" className="form-label">Confirm Password *</label>
          <input
            type="password"
            id="confirmPassword"
            className={`form-input ${validationErrors.confirmPassword ? 'error' : ''}`}
            value={formData.confirmPassword}
            onChange={(e) => handleInputChange('confirmPassword', e.target.value)}
            placeholder="Confirm your password"
            required
            disabled={isSubmitting}
          />
          {validationErrors.confirmPassword && (
            <span style={{ color: '#fecaca', fontSize: '0.8rem', marginTop: '0.25rem', display: 'block' }}>
              {validationErrors.confirmPassword}
            </span>
          )}
        </div>

        <button
          type="submit"
          className="form-button"
          disabled={isSubmitting}
        >
          {isSubmitting ? (
            <>
              <span className="loading-spinner"></span>
              Creating Account...
            </>
          ) : (
            '🎸 Create Account'
          )}
        </button>
      </form>

      <div className="auth-toggle">
        <p>
          Already have an account?{' '}
          <button
            type="button"
            onClick={onSwitchToLogin}
            disabled={isSubmitting}
          >
            Sign in here
          </button>
        </p>
      </div>
    </div>
  );
};

export default Register;
