// Chat History Storage Utilities

export interface SavedMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  enhanced?: boolean;
}

export interface SavedConversation {
  id: string;
  title: string;
  messages: SavedMessage[];
  createdAt: Date;
  lastModified: Date;
}

const STORAGE_KEY = 'atom-gpt-conversations';
const MAX_CONVERSATIONS = 50; // Limit to prevent localStorage bloat

// Get all saved conversations
export const getSavedConversations = (): SavedConversation[] => {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) return [];
    
    const conversations = JSON.parse(stored);
    // Convert date strings back to Date objects
    return conversations.map((conv: any) => ({
      ...conv,
      createdAt: new Date(conv.createdAt),
      lastModified: new Date(conv.lastModified),
      messages: conv.messages.map((msg: any) => ({
        ...msg,
        timestamp: new Date(msg.timestamp)
      }))
    }));
  } catch (error) {
    console.error('Error loading conversations:', error);
    return [];
  }
};

// Save a conversation
export const saveConversation = (conversation: SavedConversation): void => {
  try {
    const conversations = getSavedConversations();
    
    // Find existing conversation or add new one
    const existingIndex = conversations.findIndex(c => c.id === conversation.id);
    
    if (existingIndex >= 0) {
      conversations[existingIndex] = conversation;
    } else {
      conversations.unshift(conversation); // Add to beginning
    }
    
    // Limit the number of saved conversations
    const trimmedConversations = conversations.slice(0, MAX_CONVERSATIONS);
    
    localStorage.setItem(STORAGE_KEY, JSON.stringify(trimmedConversations));
  } catch (error) {
    console.error('Error saving conversation:', error);
  }
};

// Delete a conversation
export const deleteConversation = (conversationId: string): void => {
  try {
    const conversations = getSavedConversations();
    const filtered = conversations.filter(c => c.id !== conversationId);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(filtered));
  } catch (error) {
    console.error('Error deleting conversation:', error);
  }
};

// Generate a conversation title from the first user message
export const generateConversationTitle = (firstMessage: string): string => {
  const maxLength = 40;
  let title = firstMessage.trim();
  
  // Remove common prefixes
  title = title.replace(/^(tell me|what is|how do|can you|please)/i, '');
  
  // Take first sentence or first 40 characters
  const firstSentence = title.split(/[.!?]/)[0];
  title = firstSentence.length > 0 ? firstSentence : title;
  
  if (title.length > maxLength) {
    title = title.substring(0, maxLength - 3) + '...';
  }
  
  return title || 'New Conversation';
};

// Generate unique conversation ID
export const generateConversationId = (): string => {
  return `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
};

// Auto-save current conversation
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
