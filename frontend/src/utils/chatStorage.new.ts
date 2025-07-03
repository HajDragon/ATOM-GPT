/**
 * Chat Storage Utility
 * Provides localStorage-based conversation management for backward compatibility
 */

export interface SavedMessage {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: number;
  isLoading?: boolean;
}

export interface SavedConversation {
  id: string;
  title: string;
  messages: SavedMessage[];
  createdAt: Date;
  lastModified: Date;
}

export interface ChatConversation {
  id: string;
  title: string;
  messages: SavedMessage[];
  timestamp: number;
  lastActivity: number;
  isFromDatabase?: boolean;
  needsSync?: boolean;
}

export interface CompletionHistory {
  id: string;
  prompt: string;
  result: string;
  timestamp: number;
  settings: any;
  enhanced?: boolean;
  tokens?: number;
  processingTime?: number;
}

class ChatStorage {
  private readonly CHAT_STORAGE_KEY = 'atom_gpt_conversations';
  private readonly COMPLETION_STORAGE_KEY = 'atom_gpt_completions';

  // Legacy compatibility method
  saveConversationLegacy(conversation: SavedConversation): void {
    try {
      const conversations = this.getLegacyConversations();
      const existingIndex = conversations.findIndex(c => c.id === conversation.id);
      
      if (existingIndex >= 0) {
        conversations[existingIndex] = conversation;
      } else {
        conversations.unshift(conversation);
      }
      
      const trimmed = conversations.slice(0, 50);
      localStorage.setItem('atom_gpt_conversations_legacy', JSON.stringify(trimmed));
    } catch (error) {
      console.error('Error saving legacy conversation:', error);
    }
  }

  private getLegacyConversations(): SavedConversation[] {
    try {
      const stored = localStorage.getItem('atom_gpt_conversations_legacy');
      return stored ? JSON.parse(stored) : [];
    } catch (error) {
      console.error('Error loading legacy conversations:', error);
      return [];
    }
  }

  getSavedConversations(): SavedConversation[] {
    return this.getLegacyConversations();
  }

  deleteConversation(id: string): void {
    try {
      const conversations = this.getLegacyConversations();
      const filtered = conversations.filter(c => c.id !== id);
      localStorage.setItem('atom_gpt_conversations_legacy', JSON.stringify(filtered));
    } catch (error) {
      console.error('Error deleting conversation:', error);
    }
  }

  updateConversationTitle(id: string, title: string): void {
    try {
      const conversations = this.getLegacyConversations();
      const conversation = conversations.find(c => c.id === id);
      if (conversation) {
        conversation.title = title;
        localStorage.setItem('atom_gpt_conversations_legacy', JSON.stringify(conversations));
      }
    } catch (error) {
      console.error('Error updating conversation title:', error);
    }
  }
}

export const chatStorage = new ChatStorage();
export default chatStorage;

// Legacy exports for backward compatibility  
export const getSavedConversations = (): SavedConversation[] => {
  return chatStorage.getSavedConversations();
};

export const saveConversation = (conversation: SavedConversation): void => {
  chatStorage.saveConversationLegacy(conversation);
};

export const deleteConversation = (id: string): void => {
  chatStorage.deleteConversation(id);
};

export const updateConversationTitle = (id: string, title: string): void => {
  chatStorage.updateConversationTitle(id, title);
};

export const generateConversationTitle = (firstMessage: string): string => {
  if (!firstMessage) return 'New Conversation';
  
  const trimmed = firstMessage.trim();
  if (trimmed.length <= 50) return trimmed;
  
  return trimmed.slice(0, 47) + '...';
};

export const generateConversationId = (): string => {
  return 'conv_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
};

export const autoSaveConversation = (
  conversationId: string,
  messages: SavedMessage[],
  existingTitle?: string
): void => {
  if (messages.length === 0) return;
  
  const firstUserMessage = messages.find(m => m.role === 'user')?.content || '';
  const title = existingTitle || generateConversationTitle(firstUserMessage);
  
  const conversation: SavedConversation = {
    id: conversationId,
    title,
    messages,
    createdAt: new Date(messages[0]?.timestamp || Date.now()),
    lastModified: new Date()
  };
  
  saveConversation(conversation);
};
