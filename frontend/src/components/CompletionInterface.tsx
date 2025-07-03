import React, { useState, useRef, useEffect } from 'react';
import axios, { AxiosError } from 'axios';
import { api } from '../services/api';
import ModelInfo from './ModelInfo';

interface CompletionSettings {
  tokens: number;
  temperature: number;
  topP: number;
  repetitionPenalty: number;
  prompt: string;
}

const CompletionInterface: React.FC = () => {
  const [settings, setSettings] = useState<CompletionSettings>({
    tokens: 100,
    temperature: 0.8,
    topP: 0.9,
    repetitionPenalty: 1.1,
    prompt: ''
  });
  const [result, setResult] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [lmStudioEnabled, setLmStudioEnabled] = useState(true);
  const [showSettings, setShowSettings] = useState(false);
  const [wasEnhanced, setWasEnhanced] = useState(false);  // Track actual enhancement status
  const [generationProgress, setGenerationProgress] = useState(0);
  const settingsRef = useRef<HTMLDivElement>(null);
  const promptRef = useRef<HTMLTextAreaElement>(null);
  const resultRef = useRef<HTMLDivElement>(null);

  // Close settings when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (settingsRef.current && !settingsRef.current.contains(event.target as Node)) {
        setShowSettings(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const generateCompletion = async () => {
    if (!settings.prompt.trim() || isLoading) return;

    setIsLoading(true);
    setIsGenerating(true);
    setResult('');
    setGenerationProgress(0);

    try {
      const response = await api.post('/api/completion', {
        prompt: settings.prompt,
        settings: {
          tokens: settings.tokens,
          temperature: settings.temperature,
          topP: settings.topP,
          repetitionPenalty: settings.repetitionPenalty
        },
        enhance: lmStudioEnabled
      });

      // Simulate typing effect for completion
      const fullCompletion = response.data.completion;
      const words = fullCompletion.split(' ');
      let currentText = '';
      
      for (let i = 0; i < words.length; i++) {
        currentText += (i > 0 ? ' ' : '') + words[i];
        setResult(currentText);
        setGenerationProgress((i + 1) / words.length * 100);
        
        // Scroll to show new content
        if (resultRef.current) {
          resultRef.current.scrollTop = resultRef.current.scrollHeight;
        }
        
        // Add delay for typing effect
        if (i < words.length - 1) {
          await new Promise(resolve => setTimeout(resolve, 30 + Math.random() * 70));
        }
      }

      setWasEnhanced(response.data.enhanced || false);  // Track actual enhancement status

    } catch (error) {
      console.error('Error generating completion:', error);
      if (axios.isAxiosError(error) || error instanceof AxiosError) {
        if (error.response) {
          setResult(`Error ${error.response.status}: ${error.response.data?.error || 'Server error'}`);
        } else if (error.request) {
          setResult('Error: Network connection failed. Check if backend is running.');
        } else {
          setResult(`Error: ${error.message}`);
        }
      } else {
        setResult('Error: Failed to generate completion.');
      }
    } finally {
      setIsLoading(false);
      setIsGenerating(false);
      setGenerationProgress(100);
      // Focus back to prompt for better UX
      setTimeout(() => promptRef.current?.focus(), 100);
    }
  };

  const clearAll = () => {
    setSettings(prev => ({...prev, prompt: ''}));
    setResult('');
    setWasEnhanced(false);
    setGenerationProgress(0);
    setShowSettings(false);
    promptRef.current?.focus();
  };

  const copyToClipboard = async () => {
    if (result) {
      try {
        await navigator.clipboard.writeText(result);
        // Could add a toast notification here
      } catch (err) {
        console.error('Failed to copy text: ', err);
      }
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      e.preventDefault();
      generateCompletion();
    } else if (e.key === 'Escape') {
      setShowSettings(false);
    }
  };

  return (
    <div className="completion-container">
      {/* Settings Header */}
      <div className="completion-header">
        <h2>Text Completion</h2>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <ModelInfo 
            currentSettings={{
              temperature: settings.temperature,
              tokens: settings.tokens,
              topP: settings.topP,
              repetitionPenalty: settings.repetitionPenalty
            }}
            mode="Text Completion"
          />
          <div style={{ position: 'relative' }} ref={settingsRef}>
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="settings-toggle"
            title="Settings"
          >
            ⚙️
          </button>
          
          {showSettings && (
            <div className="settings-dropdown">
              <h3>Completion Settings</h3>
              
              <div className="setting-group">
                <label className="setting-label">Max Tokens</label>
                <input
                  type="number"
                  min="10"
                  max="1000"
                  value={settings.tokens}
                  onChange={(e) => setSettings(prev => ({
                    ...prev,
                    tokens: parseInt(e.target.value) || 100
                  }))}
                  className="setting-input"
                />
              </div>
              
              <div className="setting-group">
                <label className="setting-label">Temperature</label>
                <input
                  type="number"
                  min="0"
                  max="2"
                  step="0.1"
                  value={settings.temperature}
                  onChange={(e) => setSettings(prev => ({
                    ...prev,
                    temperature: parseFloat(e.target.value) || 0.8
                  }))}
                  className="setting-input"
                />
              </div>
              
              <div className="setting-group">
                <label className="setting-label">Top-p</label>
                <input
                  type="number"
                  min="0.1"
                  max="1.0"
                  step="0.1"
                  value={settings.topP}
                  onChange={(e) => setSettings(prev => ({
                    ...prev,
                    topP: parseFloat(e.target.value) || 0.9
                  }))}
                  className="setting-input"
                />
              </div>
              
              <div className="setting-group">
                <label className="setting-label">Repetition Penalty</label>
                <input
                  type="number"
                  min="1.0"
                  max="2.0"
                  step="0.1"
                  value={settings.repetitionPenalty}
                  onChange={(e) => setSettings(prev => ({
                    ...prev,
                    repetitionPenalty: parseFloat(e.target.value) || 1.1
                  }))}
                  className="setting-input"
                />
              </div>
              
              <div className="setting-checkbox">
                <input
                  type="checkbox"
                  id="lm-studio-completion-toggle"
                  checked={lmStudioEnabled}
                  onChange={(e) => setLmStudioEnabled(e.target.checked)}
                />
                <label htmlFor="lm-studio-completion-toggle">LM Studio Enhancement</label>
              </div>
              
              <button onClick={clearAll} className="clear-chat-button">
                Clear All
              </button>
            </div>
          )}
          </div>
        </div>
      </div>

      {/* Input Section */}
      <div className="prompt-container">
        <label className="setting-label" style={{ display: 'block', marginBottom: '0.75rem', fontSize: '1rem' }}>
          Enter Your Prompt {isLoading && (
            <span style={{ fontSize: '0.75rem', color: '#f59e0b' }}>
              (Generating... {Math.round(generationProgress)}%)
            </span>
          )}
        </label>
        <textarea
          ref={promptRef}
          value={settings.prompt}
          onChange={(e) => setSettings(prev => ({...prev, prompt: e.target.value}))}
          onKeyDown={handleKeyPress}
          placeholder={isLoading ? "Generating completion..." : `Enter dark metal lyrics to complete... (Ctrl+Enter to generate)

Examples:
• Through the gates of eternal darkness we march
• Beneath the blood-red moon, ancient spirits rise  
• Thunder crashes as the iron throne falls
• In the shadows of the burning cathedral
• Steel meets bone on the battlefield of souls
• The dragon's breath ignites the midnight sky
• From the depths of hell, vengeance calls
• Crimson tears fall from the weeping stone...`}
          className="prompt-input"
          style={{ minHeight: '120px' }}
          disabled={isLoading}
        />
        
        {/* Example Buttons */}
        <div style={{ marginTop: '0.75rem', display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
          <button
            onClick={() => setSettings(prev => ({...prev, prompt: 'Through the gates of eternal darkness we march'}))}
            className="button-secondary"
            style={{ fontSize: '0.75rem', padding: '0.25rem 0.5rem' }}
            disabled={isLoading}
          >
            🚪 Gates of Darkness
          </button>
          <button
            onClick={() => setSettings(prev => ({...prev, prompt: 'Beneath the blood-red moon, ancient spirits rise'}))}
            className="button-secondary"
            style={{ fontSize: '0.75rem', padding: '0.25rem 0.5rem' }}
            disabled={isLoading}
          >
            🌙 Blood Moon
          </button>
          <button
            onClick={() => setSettings(prev => ({...prev, prompt: 'Thunder crashes as the iron throne falls'}))}
            className="button-secondary"
            style={{ fontSize: '0.75rem', padding: '0.25rem 0.5rem' }}
            disabled={isLoading}
          >
            ⚡ Iron Throne
          </button>
          <button
            onClick={() => setSettings(prev => ({...prev, prompt: 'In the shadows of the burning cathedral'}))}
            className="button-secondary"
            style={{ fontSize: '0.75rem', padding: '0.25rem 0.5rem' }}
            disabled={isLoading}
          >
            🔥 Burning Cathedral
          </button>
          <button
            onClick={() => setSettings(prev => ({...prev, prompt: 'Steel meets bone on the battlefield of souls'}))}
            className="button-secondary"
            style={{ fontSize: '0.75rem', padding: '0.25rem 0.5rem' }}
            disabled={isLoading}
          >
            ⚔️ Steel & Bone
          </button>
          <button
            onClick={() => setSettings(prev => ({...prev, prompt: 'From the depths of hell, vengeance calls'}))}
            className="button-secondary"
            style={{ fontSize: '0.75rem', padding: '0.25rem 0.5rem' }}
            disabled={isLoading}
          >
            🔥 Hell's Vengeance
          </button>
        </div>
        
        <div className="button-group">
          <button
            onClick={generateCompletion}
            disabled={!settings.prompt.trim() || isLoading}
            className="generate-button"
          >
            {isLoading ? (
              <>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <div className="dot" style={{ animation: 'typingAnimation 1.4s infinite ease-in-out' }}></div>
                  Generating...
                </div>
              </>
            ) : (
              '✨ Generate Completion'
            )}
          </button>
          <button
            onClick={clearAll}
            className="button-secondary"
          >
            🗑️ Clear All
          </button>
        </div>
      </div>

      {/* Output Section */}
      <div className="completion-output" ref={resultRef}>
        {isLoading && !result ? (
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            height: '100%',
            flexDirection: 'column',
            gap: '1rem'
          }}>
            <div className="typing-dots">
              <div className="dot"></div>
              <div className="dot"></div>
              <div className="dot"></div>
            </div>
            <span style={{ color: '#8e8ea0' }}>ATOM-GPT is generating your completion...</span>
            {generationProgress > 0 && (
              <div style={{ width: '100%', maxWidth: '200px' }}>
                <div style={{ 
                  width: '100%', 
                  height: '4px', 
                  backgroundColor: '#444', 
                  borderRadius: '2px',
                  overflow: 'hidden'
                }}>
                  <div style={{ 
                    width: `${generationProgress}%`, 
                    height: '100%', 
                    backgroundColor: '#dc143c',
                    transition: 'width 0.3s ease'
                  }} />
                </div>
                <span style={{ fontSize: '0.75rem', color: '#8e8ea0' }}>
                  {Math.round(generationProgress)}%
                </span>
              </div>
            )}
          </div>
        ) : result ? (
          <div>
            <div style={{ 
              borderBottom: '1px solid #444', 
              paddingBottom: '0.75rem',
              marginBottom: '1rem',
              fontSize: '0.875rem',
              color: '#b0b0b0',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              flexWrap: 'wrap',
              gap: '0.5rem'
            }}>
              <span>
                <strong>Completion Result:</strong> {result.length} characters
              </span>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <button
                  onClick={copyToClipboard}
                  className="button-secondary"
                  style={{ fontSize: '0.75rem', padding: '0.25rem 0.5rem' }}
                  title="Copy to clipboard"
                >
                  📋 Copy
                </button>
                {wasEnhanced && (
                  <span className="enhanced-badge">
                    ✨ Enhanced by LM Studio
                  </span>
                )}
              </div>
            </div>
            <div style={{ position: 'relative' }}>
              {result}
              {isGenerating && (
                <span className="typing-cursor" style={{ 
                  animation: 'blink 1s infinite',
                  marginLeft: '2px'
                }}>|</span>
              )}
            </div>
          </div>
        ) : (
          <div className="empty-state">
            <h3>Ready to Generate</h3>
            <p>Enter a prompt above and click "Generate Completion" to get started with ATOM-GPT.</p>
          </div>
        )}
      </div>

      {/* Metal Examples Section */}
      <div className="prompt-container">
        <h3 style={{ marginBottom: '1rem', color: '#e0e0e0', fontWeight: '600' }}>🔥 Metal Lyric Examples</h3>
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
          gap: '0.75rem'
        }}>
          {[
            "Through the gates of eternal darkness we march, swords gleaming with unholy light",
            "Beneath the blood-red moon, ancient spirits rise from forgotten tombs",
            "Thunder crashes as the iron throne falls, kingdoms crumble to dust",
            "In the shadows of the burning cathedral, prayers turn to screams",
            "Steel meets bone on the battlefield of souls, where heroes become legends",
            "The dragon's breath ignites the midnight sky, painting clouds in crimson fire"
          ].map((example, index) => (
            <button
              key={index}
              onClick={() => setSettings(prev => ({...prev, prompt: example}))}
              style={{
                padding: '1rem',
                background: '#2f2f2f',
                border: '1px solid #444',
                borderRadius: '8px',
                textAlign: 'left',
                fontSize: '0.875rem',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                color: '#e0e0e0'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = '#3a3a3a';
                e.currentTarget.style.borderColor = '#dc143c';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = '#2f2f2f';
                e.currentTarget.style.borderColor = '#444';
              }}
            >
              {example}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default CompletionInterface;
