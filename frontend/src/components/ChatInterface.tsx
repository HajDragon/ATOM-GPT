import React, { useState, useEffect, useRef } from 'react';
import { api } from '../services/api';
import ModelInfo from './ModelInfo';
import { 
  SavedConversation, 
  SavedMessage, 
  autoSaveConversation, 
  generateConversationId,
  generateConversationTitle
} from '../utils/chatStorage';

interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  enhanced?: boolean;
}

interface ChatSettings {
  tokens: number;
  temperature: number;
  topP: number;
  repetitionPenalty: number;
}

interface ChatInterfaceProps {
  selectedConversation?: SavedConversation | null;
  newChatTrigger?: number;
  onConversationUpdate?: (conversationId: string) => void;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ 
  selectedConversation, 
  newChatTrigger, 
  onConversationUpdate 
}) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [settings, setSettings] = useState<ChatSettings>({
    tokens: 100,
    temperature: 0.8,
    topP: 0.9,
    repetitionPenalty: 1.1
  });
  const [lmStudioEnabled, setLmStudioEnabled] = useState(true);
  const [currentConversationId, setCurrentConversationId] = useState<string>(generateConversationId());
  const [conversationTitle, setConversationTitle] = useState<string>('');
  const [messageBuffer, setMessageBuffer] = useState<string>('');
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const settingsRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Auto-save conversation when messages change
  useEffect(() => {
    if (messages.length > 0) {
      const savedMessages = messages.map(msg => ({
        ...msg,
        timestamp: msg.timestamp instanceof Date ? msg.timestamp.getTime() : msg.timestamp
      })) as SavedMessage[];
      autoSaveConversation(currentConversationId, savedMessages, conversationTitle);
    }
  }, [messages, currentConversationId, conversationTitle]);

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

  // Load selected conversation
  useEffect(() => {
    if (selectedConversation) {
      const convertedMessages: Message[] = selectedConversation.messages.map(msg => ({
        ...msg,
        timestamp: new Date(msg.timestamp)
      }));
      setMessages(convertedMessages);
      setCurrentConversationId(selectedConversation.id);
      setConversationTitle(selectedConversation.title);
    }
  }, [selectedConversation]);

  // Handle new chat trigger
  useEffect(() => {
    if (newChatTrigger && newChatTrigger > 0) {
      handleNewChat();
    }
  }, [newChatTrigger]);

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
      timestamp: new Date()
    };

    // Set conversation title from first message if not set
    if (messages.length === 0 && !conversationTitle) {
      setConversationTitle(generateConversationTitle(input.trim()));
    }

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    setIsTyping(true);

    // Create a temporary assistant message for streaming
    const tempMessageId = (Date.now() + 1).toString();
    setStreamingMessageId(tempMessageId);
    setMessageBuffer('');
    
    const tempAssistantMessage: Message = {
      id: tempMessageId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      enhanced: false
    };
    
    setMessages(prev => [...prev, tempAssistantMessage]);

    try {
      const response = await api.post('/api/chat', {
        message: userMessage.content,
        settings: settings,
        enhance: lmStudioEnabled,
        history: messages.slice(-10) // Send last 10 messages for context
      });

      // Simulate typing effect for better UX
      const fullResponse = response.data.response;
      const words = fullResponse.split(' ');
      let currentText = '';
      
      for (let i = 0; i < words.length; i++) {
        currentText += (i > 0 ? ' ' : '') + words[i];
        
        setMessages(prev => prev.map(msg => 
          msg.id === tempMessageId 
            ? { ...msg, content: currentText, enhanced: response.data.enhanced }
            : msg
        ));
        
        // Add delay for typing effect
        if (i < words.length - 1) {
          await new Promise(resolve => setTimeout(resolve, 50 + Math.random() * 100));
        }
      }

      setStreamingMessageId(null);

    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: Message = {
        id: tempMessageId,
        role: 'system',
        content: 'Error: Failed to get response from the model.',
        timestamp: new Date()
      };
      setMessages(prev => prev.map(msg => 
        msg.id === tempMessageId ? errorMessage : msg
      ));
      setStreamingMessageId(null);
    } finally {
      setIsLoading(false);
      setIsTyping(false);
      // Focus back to input for better UX
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  };

  const clearChat = () => {
    handleNewChat(); // Use the new chat handler instead
    setShowSettings(false);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    } else if (e.key === 'Escape') {
      setShowSettings(false);
    }
  };

  const adjustTextareaHeight = (element: HTMLTextAreaElement) => {
    element.style.height = '48px';
    const scrollHeight = element.scrollHeight;
    const maxHeight = 120; // Max 3 lines roughly
    element.style.height = Math.min(scrollHeight, maxHeight) + 'px';
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    adjustTextareaHeight(e.target);
  };

  const handleNewChat = () => {
    setMessages([]);
    setInput('');
    const newId = generateConversationId();
    setCurrentConversationId(newId);
    setConversationTitle('');
    onConversationUpdate?.(newId);
  };

  return (
    <div className="chat-container">
      {/* Settings Toggle and Model Info */}
      <div style={{ position: 'absolute', top: '1rem', right: '1rem', zIndex: 1001 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <ModelInfo 
            currentSettings={{
              temperature: settings.temperature,
              tokens: settings.tokens,
              topP: settings.topP,
              repetitionPenalty: settings.repetitionPenalty
            }}
            mode="Normal Chat"
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
              <h3>Settings</h3>
              
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
                  id="lm-studio-toggle"
                  checked={lmStudioEnabled}
                  onChange={(e) => setLmStudioEnabled(e.target.checked)}
                />
                <label htmlFor="lm-studio-toggle">LM Studio Enhancement</label>
              </div>
              
              <button onClick={clearChat} className="clear-chat-button">
                Clear Conversation
              </button>
            </div>
          )}
          </div>
        </div>
      </div>

      {/* Messages Container */}
      <div className="messages-container">
        {messages.length === 0 ? (
          <div className="empty-state">
            <h3>Welcome to ATOM-GPT</h3>
            <p>Start a conversation and your messages will be enhanced by LM Studio when available.</p>
          </div>
        ) : (
          <>
            {messages.map((message) => (
              <div key={message.id} className={`message ${message.role}`}>
                <div className="message-avatar">
                  {message.role === 'user' ? 'U' : message.role === 'assistant' ? 'A' : 'S'}
                </div>
                <div className="message-content">
                  {message.content}
                  {message.id === streamingMessageId && isTyping && (
                    <span className="typing-cursor">|</span>
                  )}
                  {message.enhanced && !isTyping && (
                    <span className="enhanced-badge">
                      ✨ Enhanced
                    </span>
                  )}
                </div>
              </div>
            ))}
            {isLoading && !streamingMessageId && (
              <div className="typing-indicator">
                <div className="message-avatar" style={{ backgroundColor: '#ab68ff', color: 'white' }}>
                  A
                </div>
                <div className="typing-dots">
                  <div className="dot"></div>
                  <div className="dot"></div>
                  <div className="dot"></div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* Input Area */}
      <div className="input-container">
        <div className="input-wrapper">
          <textarea
            ref={inputRef}
            value={input}
            onChange={handleInputChange}
            onKeyDown={handleKeyPress}
            placeholder={isLoading ? "Generating response..." : "Message ATOM-GPT... (Enter to send, Shift+Enter for new line)"}
            className="message-input"
            disabled={isLoading}
            rows={1}
            style={{ resize: 'none' }}
          />
          <button
            onClick={sendMessage}
            disabled={!input.trim() || isLoading}
            className="send-button"
            title="Send message"
          >
            {isLoading ? (
              <div className="spinner" style={{ 
                width: '12px', 
                height: '12px', 
                border: '2px solid #444',
                borderTop: '2px solid #fff',
                borderRadius: '50%',
                animation: 'spin 1s linear infinite'
              }} />
            ) : (
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path
                  d="M.5 1.163A1 1 0 0 1 1.97.28l12.868 6.837a1 1 0 0 1 0 1.766L1.969 15.72A1 1 0 0 1 .5 14.836V10.33a1 1 0 0 1 .816-.983L8.5 8 1.316 6.653A1 1 0 0 1 .5 5.67V1.163Z"
                  fill="currentColor"
                />
              </svg>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
