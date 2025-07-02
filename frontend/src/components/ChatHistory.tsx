import React, { useState, useEffect, useRef } from 'react';
import { SavedConversation, getSavedConversations, deleteConversation } from '../utils/chatStorage';

interface ChatHistoryProps {
  onSelectConversation: (conversation: SavedConversation) => void;
  onNewChat: () => void;
  currentConversationId?: string;
}

const ChatHistory: React.FC<ChatHistoryProps> = ({ 
  onSelectConversation, 
  onNewChat, 
  currentConversationId 
}) => {
  const [conversations, setConversations] = useState<SavedConversation[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const historyRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    loadConversations();
  }, []);

  // Close history when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (historyRef.current && !historyRef.current.contains(event.target as Node)) {
        setShowHistory(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const loadConversations = () => {
    const saved = getSavedConversations();
    setConversations(saved);
  };

  const handleDeleteConversation = (conversationId: string, event: React.MouseEvent) => {
    event.stopPropagation();
    if (window.confirm('Are you sure you want to delete this conversation?')) {
      deleteConversation(conversationId);
      loadConversations();
    }
  };

  const filteredConversations = conversations.filter(conv =>
    conv.title.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const formatDate = (date: Date) => {
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));
    
    if (days === 0) return 'Today';
    if (days === 1) return 'Yesterday';
    if (days < 7) return `${days} days ago`;
    return date.toLocaleDateString();
  };

  return (
    <div className="chat-history-container" ref={historyRef}>
      {/* History Toggle Button */}
      <button
        onClick={() => setShowHistory(!showHistory)}
        className="history-toggle"
        title="Chat History"
      >
        📋
      </button>

      {/* History Panel */}
      {showHistory && (
        <div className="history-panel">
          <div className="history-header">
            <h3>Chat History</h3>
            <button onClick={onNewChat} className="new-chat-button">
              ➕ New Chat
            </button>
          </div>

          {/* Search */}
          <div className="history-search">
            <input
              type="text"
              placeholder="Search conversations..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="search-input"
            />
          </div>

          {/* Conversations List */}
          <div className="conversations-list">
            {filteredConversations.length === 0 ? (
              <div className="empty-history">
                <p>No conversations found</p>
                <button onClick={onNewChat} className="button-secondary">
                  Start Your First Chat
                </button>
              </div>
            ) : (
              filteredConversations.map((conversation) => (
                <div
                  key={conversation.id}
                  className={`conversation-item ${
                    conversation.id === currentConversationId ? 'active' : ''
                  }`}
                  onClick={() => {
                    onSelectConversation(conversation);
                    setShowHistory(false);
                  }}
                >
                  <div className="conversation-content">
                    <div className="conversation-title">{conversation.title}</div>
                    <div className="conversation-meta">
                      <span className="conversation-date">
                        {formatDate(conversation.lastModified)}
                      </span>
                      <span className="conversation-messages">
                        {conversation.messages.length} messages
                      </span>
                    </div>
                  </div>
                  <button
                    onClick={(e) => handleDeleteConversation(conversation.id, e)}
                    className="delete-conversation"
                    title="Delete conversation"
                  >
                    🗑️
                  </button>
                </div>
              ))
            )}
          </div>

          {/* Storage Info */}
          <div className="storage-info">
            <small>
              {conversations.length} conversation{conversations.length !== 1 ? 's' : ''} saved locally
            </small>
          </div>
        </div>
      )}
    </div>
  );
};

export default ChatHistory;
