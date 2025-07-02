import React from 'react';
import { Link, useLocation } from 'react-router-dom';

const Navigation: React.FC = () => {
  const location = useLocation();

  const isActive = (path: string) => {
    return location.pathname === path;
  };

  const navItems = [
    { path: '/chat', label: 'Chat', description: 'Interactive conversation mode' },
    { path: '/completion', label: 'Completion', description: 'Text completion mode' }
  ];

  return (
    <nav className="navigation">
      {navItems.map((item) => (
        <Link
          key={item.path}
          to={item.path}
          className={`nav-button ${isActive(item.path) ? 'active' : ''}`}
        >
          <div>
            <div>{item.label}</div>
            <div style={{ fontSize: '0.8rem', opacity: 0.8 }}>{item.description}</div>
          </div>
        </Link>
      ))}
    </nav>
  );
};

export default Navigation;
