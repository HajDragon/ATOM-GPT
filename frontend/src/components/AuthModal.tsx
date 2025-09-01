import React, { useState } from 'react';
import Login from './Login';
import Register from './Register';

interface AuthModalProps {
  isOpen: boolean;
  onClose: () => void;
  initialMode?: 'login' | 'register';
  showCloseButton?: boolean;
}

const AuthModal: React.FC<AuthModalProps> = ({
  isOpen,
  onClose,
  initialMode = 'login',
  showCloseButton = true
}) => {
  const [mode, setMode] = useState<'login' | 'register'>(initialMode);

  if (!isOpen) return null;

  const handleOverlayClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget && showCloseButton) {
      onClose();
    }
  };

  return (
    <div className="modal-overlay" onClick={handleOverlayClick}>
      <div className="modal-content">
        {showCloseButton && (
          <button className="modal-close" onClick={onClose}>
            ×
          </button>
        )}

        {mode === 'login' ? (
          <Login
            onSwitchToRegister={() => setMode('register')}
            onClose={onClose}
          />
        ) : (
          <Register
            onSwitchToLogin={() => setMode('login')}
            onClose={onClose}
          />
        )}
      </div>
    </div>
  );
};

export default AuthModal;
