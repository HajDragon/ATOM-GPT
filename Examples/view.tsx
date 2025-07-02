import React, { useState, useEffect, useRef } from 'react';
import './App.css';

const App = () => {
  const [messages, setMessages] = useState([
    { text: 'Hello! I am AtomGPT. How can I assist you today?', sender: 'ai' }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = (e) => {
    e.preventDefault();
    if (!inputValue.trim()) return;

    // Add user message
    const newMessages = [...messages, { text: inputValue, sender: 'user' }];
    setMessages(newMessages);
    setInputValue('');
    setIsTyping(true);

    // Simulate AI response after a delay
    setTimeout(() => {
      const aiResponses = [
        "I understand your question about that topic.",
        "That's an interesting point you've raised.",
        "Let me think about that for a moment...",
        "Based on my knowledge, I'd suggest considering multiple perspectives.",
        "I can provide more details if you'd like."
      ];
      const randomResponse = aiResponses[Math.floor(Math.random() * aiResponses.length)];
      setMessages([...newMessages, { text: randomResponse, sender: 'ai' }]);
      setIsTyping(false);
    }, 1500);
  };

  return (
    <div className="app-container">
      <div className="header">
        <div className="logo-container">
          <div className="logo-circle">
            <span className="logo-text">A</span>
          </div>
          <h1>AtomGPT</h1>
        </div>
        <div className="header-tabs">
          <button className="tab active">Chat</button>
          <button className="tab">History</button>
          <button className="tab">Model Details</button>
          <button className="tab">Instructions</button>
        </div>
        <div className="mode-selector">
          <select>
            <option>Standard Mode</option>
            <option>Creative Mode</option>
            <option>Precise Mode</option>
          </select>
        </div>
      </div>

      <div className="chat-container">
        <div className="messages-container">
          {messages.map((message, index) => (
            <div 
              key={index} 
              className={`message ${message.sender}`}
              style={{
                background: message.sender === 'ai' 
                  ? 'linear-gradient(135deg, #e6f2ff, #cce5ff)' 
                  : 'linear-gradient(135deg, #0077cc, #005fa3)',
                alignSelf: message.sender === 'ai' ? 'flex-start' : 'flex-end'
              }}
            >
              {message.text}
            </div>
          ))}
          {isTyping && (
            <div className="message ai typing-indicator">
              <div className="typing-dots">
                <div className="dot"></div>
                <div className="dot"></div>
                <div className="dot"></div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <form onSubmit={handleSendMessage} className="input-container">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Type your message here..."
            className="message-input"
          />
          <button type="submit" className="send-button">
            <svg viewBox="0 0 24 24" width="24" height="24">
              <path fill="white" d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"></path>
            </svg>
          </button>
        </form>
      </div>
    </div>
  );
};

export default App;