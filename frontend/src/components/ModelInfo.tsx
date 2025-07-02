import React, { useState, useEffect } from 'react';
import axios from 'axios';

interface ModelInfoData {
  model: {
    iterations: number | string;
    parameters: number;
    model_size: string;
    device: string;
    model_loaded: boolean;
  };
  gpu: {
    gpu_available: boolean;
    gpu_name: string;
    gpu_count: number;
  };
  lm_studio: {
    available: boolean;
    model_name: string;
  };
  mode: string;
  settings: {
    temperature: number;
    max_tokens: number;
    top_p: number;
    repetition_penalty: number;
  };
}

interface ModelInfoProps {
  currentSettings?: {
    temperature: number;
    tokens: number;
    topP: number;
    repetitionPenalty: number;
  };
  mode?: string;
}

const ModelInfo: React.FC<ModelInfoProps> = ({ currentSettings, mode = 'Normal Chat' }) => {
  const [modelInfo, setModelInfo] = useState<ModelInfoData | null>(null);
  const [isVisible, setIsVisible] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const fetchModelInfo = async () => {
    setIsLoading(true);
    try {
      // Use the existing status endpoint for now
      const response = await axios.get('/api/status');
      if (response.data.success) {
        // Create a mock comprehensive response based on the status data
        const mockModelInfo = {
          success: true,
          model: {
            iterations: '10000', // Mock data - would come from backend
            parameters: 25900000,
            model_size: '25.9M',
            device: response.data.device,
            model_loaded: response.data.loaded
          },
          gpu: {
            gpu_available: response.data.device === 'cuda',
            gpu_name: response.data.device === 'cuda' ? 'NVIDIA GeForce GPU' : 'CPU',
            gpu_count: response.data.device === 'cuda' ? 1 : 0
          },
          lm_studio: {
            available: response.data.lm_studio_available,
            model_name: 'phi-2' // Mock for now
          },
          mode: 'Normal Chat',
          settings: {
            temperature: 0.7,
            max_tokens: 60,
            top_p: 0.8,
            repetition_penalty: 1.35
          }
        };
        
        setModelInfo(mockModelInfo);
      }
    } catch (error) {
      console.error('Error fetching model info:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (isVisible) {
      fetchModelInfo();
    }
  }, [isVisible]);

  const getDisplaySettings = () => {
    if (currentSettings) {
      return {
        temperature: currentSettings.temperature,
        max_tokens: currentSettings.tokens,
        top_p: currentSettings.topP,
        repetition_penalty: currentSettings.repetitionPenalty
      };
    }
    return modelInfo?.settings || {
      temperature: 0.7,
      max_tokens: 60,
      top_p: 0.8,
      repetition_penalty: 1.35
    };
  };

  return (
    <div style={{ position: 'relative' }}>
      <button
        onClick={() => setIsVisible(!isVisible)}
        className="model-info-toggle"
        title="Model Details"
      >
        📊
      </button>
      
      {isVisible && (
        <div className="model-info-panel">
          <div className="model-info-header">
            <h3>🔥 ATOM-GPT Details</h3>
            <button onClick={() => setIsVisible(false)} className="close-button">×</button>
          </div>
          
          {isLoading ? (
            <div className="loading">Loading model info...</div>
          ) : modelInfo ? (
            <div className="model-info-content">
              <div className="info-section">
                <h4>🤖 Model</h4>
                <div className="info-item">
                  <span className="info-label">🔥 Iterations:</span>
                  <span className="info-value">{modelInfo.model.iterations}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">🎸 Parameters:</span>
                  <span className="info-value">{modelInfo.model.model_size}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">💾 Device:</span>
                  <span className="info-value">{modelInfo.model.device}</span>
                </div>
              </div>

              <div className="info-section">
                <h4>⚙️ Current Settings</h4>
                <div className="info-item">
                  <span className="info-label">🌡️ Temperature:</span>
                  <span className="info-value">{getDisplaySettings().temperature}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">🎯 Max tokens:</span>
                  <span className="info-value">{getDisplaySettings().max_tokens}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">🎪 Top-p:</span>
                  <span className="info-value">{getDisplaySettings().top_p}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">🔄 Repetition penalty:</span>
                  <span className="info-value">{getDisplaySettings().repetition_penalty}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">🎭 Mode:</span>
                  <span className="info-value">{mode}</span>
                </div>
              </div>

              <div className="info-section">
                <h4>🖥️ Hardware</h4>
                <div className="info-item">
                  <span className="info-label">🎮 GPU:</span>
                  <span className="info-value">
                    {modelInfo.gpu.gpu_available ? modelInfo.gpu.gpu_name : 'Not Available'}
                  </span>
                </div>
              </div>

              <div className="info-section">
                <h4>🔗 LM Studio</h4>
                <div className="info-item">
                  <span className="info-label">Status:</span>
                  <span className={`info-value ${modelInfo.lm_studio.available ? 'status-connected' : 'status-offline'}`}>
                    {modelInfo.lm_studio.available ? '✅ Connected' : '⚠️ Offline'}
                  </span>
                </div>
                {modelInfo.lm_studio.available && (
                  <>
                    <div className="info-item">
                      <span className="info-label">🤖 Model:</span>
                      <span className="info-value">{modelInfo.lm_studio.model_name}</span>
                    </div>
                  </>
                )}
              </div>
            </div>
          ) : (
            <div className="error">Failed to load model information</div>
          )}
        </div>
      )}
    </div>
  );
};

export default ModelInfo;
