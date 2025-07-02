import React, { useState } from 'react';
import ChatInterface from './ChatInterface';
import ChatSidebar from './ChatSidebar';
import Navigation from './Navigation';
import StatusPanel from './StatusPanel';
import { SavedConversation } from '../utils/chatStorage';

interface ChatLayoutProps {
  lmStudioStatus: 'connected' | 'disconnected' | 'checking';
  modelStatus: 'loaded' | 'loading' | 'error';
  onRefresh: () => void;
}

const ChatLayout: React.FC<ChatLayoutProps> = ({ lmStudioStatus, modelStatus, onRefresh }) => {
  const [selectedConversation, setSelectedConversation] = useState<SavedConversation | null>(null);
  const [newChatTrigger, setNewChatTrigger] = useState(0);

  const handleSelectConversation = (conversation: SavedConversation) => {
    setSelectedConversation(conversation);
  };

  const handleNewChat = () => {
    setSelectedConversation(null);
    setNewChatTrigger(prev => prev + 1); // Trigger re-render in ChatInterface
  };

  return (
    <>
      <ChatSidebar
        onSelectConversation={handleSelectConversation}
        onNewChat={handleNewChat}
        currentConversationId={selectedConversation?.id}
      />
      <div className="main-content">
        <header className="header">
          <div className="logo-container">
            <div className="logo-circle">A</div>
            <div className="logo-text">ATOM-GPT</div>
          </div>
          
          <div className="header-controls">
            <Navigation />
            <StatusPanel 
              lmStudioStatus={lmStudioStatus}
              modelStatus={modelStatus}
              onRefresh={onRefresh}
            />
          </div>
        </header>
        <ChatInterface 
          selectedConversation={selectedConversation}
          newChatTrigger={newChatTrigger}
          onConversationUpdate={(conversationId) => {
            // Update sidebar when conversation changes
            if (selectedConversation?.id !== conversationId) {
              setSelectedConversation(null);
            }
          }}
        />
      </div>
    </>
  );
};

export default ChatLayout;
