import React, { useState, useEffect } from 'react';
import { SavedConversation, getSavedConversations, deleteConversation } from '../utils/chatStorage';

interface ChatSidebarProps {
  onSelectConversation: (conversation: SavedConversation) => void;
  onNewChat: () => void;
  currentConversationId?: string;
}

const ChatSidebar: React.FC<ChatSidebarProps> = ({ 
  onSelectConversation, 
  onNewChat, 
  currentConversationId 
}) => {
  const [conversations, setConversations] = useState<SavedConversation[]>([]);
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    loadConversations();
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
    if (days < 7) return `${days}d ago`;
    if (days < 30) return `${Math.floor(days / 7)}w ago`;
    return date.toLocaleDateString();
  };

  return (
    <div className="chat-sidebar">
      {/* Sidebar Header */}
      <div className="sidebar-header">
        <h3 className="sidebar-title">Chat History</h3>
        <button onClick={onNewChat} className="new-chat-sidebar-button">
          ➕ New Chat
        </button>
      </div>

      {/* Search */}
      <div className="sidebar-search">
        <input
          type="text"
          placeholder="Search conversations..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="sidebar-search-input"
        />
      </div>

      {/* Conversations List */}
      <div className="sidebar-conversations">
        {filteredConversations.length === 0 ? (
          <div className="sidebar-empty-state">
            <p>No conversations found</p>
            <button onClick={onNewChat} className="button-secondary">
              Start Your First Chat
            </button>
          </div>
        ) : (
          filteredConversations.map((conversation) => (
            <div
              key={conversation.id}
              className={`sidebar-conversation-item ${
                conversation.id === currentConversationId ? 'active' : ''
              }`}
              onClick={() => onSelectConversation(conversation)}
            >
              <div className="sidebar-conversation-content">
                <div className="sidebar-conversation-title">{conversation.title}</div>
                <div className="sidebar-conversation-meta">
                  <span>{formatDate(conversation.lastModified)}</span>
                  <span>{conversation.messages.length} msgs</span>
                </div>
              </div>
              <button
                onClick={(e) => handleDeleteConversation(conversation.id, e)}
                className="sidebar-delete-button"
                title="Delete conversation"
              >
                🗑️
              </button>
            </div>
          ))
        )}
      </div>

      {/* Footer */}
      <div className="sidebar-footer">
        <small>
          {conversations.length} conversation{conversations.length !== 1 ? 's' : ''} saved
        </small>
      </div>
    </div>
  );
};

export default ChatSidebar;
