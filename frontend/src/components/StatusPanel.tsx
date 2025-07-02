import React from 'react';

interface StatusPanelProps {
  lmStudioStatus: 'connected' | 'disconnected' | 'checking';
  modelStatus: 'loaded' | 'loading' | 'error';
  onRefresh: () => void;
}

const StatusPanel: React.FC<StatusPanelProps> = ({
  lmStudioStatus,
  modelStatus,
  onRefresh
}) => {
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
    </div>
  );
};

export default StatusPanel;
