import { authApi } from './api';
import authService from './auth';

const API_BASE_URL = ''; // Using authApi baseURL

export interface Conversation {
  id: string;
  user_id: number;
  title: string;
  created_at: string;
  updated_at: string;
  last_message_at?: string;
  message_count: number;
  is_archived: boolean;
  conversation_type: 'chat' | 'completion';
}

export interface Message {
  id: string;
  conversation_id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  enhanced: boolean;
  tokens_used?: number;
  processing_time?: number;
  model_used?: string;
  temperature?: number;
  top_p?: number;
  repetition_penalty?: number;
  prompt_tokens?: number;
  completion_tokens?: number;
  created_at: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
  enhanced?: boolean;
  tokens?: number;
  processingTime?: number;
}

class DatabaseService {
  private ensureAuth() {
    if (!authService.isAuthenticated()) {
      throw new Error('User not authenticated');
    }
  }

  // Conversation Management
  async createConversation(title: string, type: 'chat' | 'completion' = 'chat'): Promise<string> {
    this.ensureAuth();
    try {
      const response = await authApi.post('/conversations', {
        title,
        conversation_type: type
      });
      return response.data.conversation_id;
    } catch (error: any) {
      throw new Error(error.response?.data?.error || 'Failed to create conversation');
    }
  }

  async getConversations(type?: 'chat' | 'completion', limit: number = 50): Promise<Conversation[]> {
    this.ensureAuth();
    try {
      const params: any = { limit };
      if (type) params.type = type;

      const response = await authApi.get('/conversations', { params });
      return response.data.conversations;
    } catch (error: any) {
      throw new Error(error.response?.data?.error || 'Failed to get conversations');
    }
  }

  async getConversation(conversationId: string): Promise<Conversation> {
    this.ensureAuth();
    try {
      const response = await authApi.get(`/conversations/${conversationId}`);
      return response.data.conversation;
    } catch (error: any) {
      throw new Error(error.response?.data?.error || 'Failed to get conversation');
    }
  }

  async deleteConversation(conversationId: string): Promise<void> {
    this.ensureAuth();
    try {
      await authApi.delete(`/conversations/${conversationId}`);
    } catch (error: any) {
      throw new Error(error.response?.data?.error || 'Failed to delete conversation');
    }
  }

  async updateConversationTitle(conversationId: string, title: string): Promise<void> {
    this.ensureAuth();
    try {
      await authApi.patch(`/conversations/${conversationId}`, { title });
    } catch (error: any) {
      throw new Error(error.response?.data?.error || 'Failed to update conversation title');
    }
  }

  // Message Management
  async addMessage(
    conversationId: string,
    role: 'user' | 'assistant',
    content: string,
    metadata?: {
      enhanced?: boolean;
      tokens_used?: number;
      processing_time?: number;
      model_used?: string;
      temperature?: number;
      top_p?: number;
      repetition_penalty?: number;
    }
  ): Promise<string> {
    this.ensureAuth();
    try {
      const response = await authApi.post(`/conversations/${conversationId}/messages`, {
        role,
        content,
        ...metadata
      });
      return response.data.message_id;
    } catch (error: any) {
      throw new Error(error.response?.data?.error || 'Failed to add message');
    }
  }

  async getMessages(conversationId: string): Promise<Message[]> {
    this.ensureAuth();
    try {
      const response = await authApi.get(`/conversations/${conversationId}/messages`);
      return response.data.messages;
    } catch (error: any) {
      throw new Error(error.response?.data?.error || 'Failed to get messages');
    }
  }

  // Convert between database messages and chat messages
  dbMessageToChatMessage(dbMessage: Message): ChatMessage {
    return {
      id: dbMessage.id,
      role: dbMessage.role as 'user' | 'assistant',
      content: dbMessage.content,
      timestamp: new Date(dbMessage.created_at).getTime(),
      enhanced: dbMessage.enhanced,
      tokens: dbMessage.tokens_used,
      processingTime: dbMessage.processing_time
    };
  }

  chatMessageToDbMessage(chatMessage: ChatMessage, conversationId: string): Partial<Message> {
    return {
      conversation_id: conversationId,
      role: chatMessage.role,
      content: chatMessage.content,
      enhanced: chatMessage.enhanced || false,
      tokens_used: chatMessage.tokens,
      processing_time: chatMessage.processingTime
    };
  }

  // Settings Management (delegated to auth service)
  async getSettings(): Promise<Record<string, any>> {
    return authService.getUserSettings();
  }

  async updateSettings(settings: Record<string, any>): Promise<Record<string, any>> {
    return authService.updateUserSettings(settings);
  }

  // Sync local storage with database
  async syncConversationToDatabase(
    localConversation: any,
    conversationType: 'chat' | 'completion' = 'chat'
  ): Promise<string> {
    try {
      // Create conversation in database
      const conversationId = await this.createConversation(
        localConversation.title || 'Untitled Conversation',
        conversationType
      );

      // Add all messages to database
      for (const message of localConversation.messages || []) {
        await this.addMessage(
          conversationId,
          message.role,
          message.content,
          {
            enhanced: message.enhanced,
            tokens_used: message.tokens,
            processing_time: message.processingTime
          }
        );
      }

      return conversationId;
    } catch (error: any) {
      throw new Error(`Failed to sync conversation: ${error.message}`);
    }
  }

  async loadConversationFromDatabase(conversationId: string): Promise<{
    conversation: Conversation;
    messages: ChatMessage[];
  }> {
    try {
      const [conversation, messages] = await Promise.all([
        this.getConversation(conversationId),
        this.getMessages(conversationId)
      ]);

      const chatMessages = messages.map(msg => this.dbMessageToChatMessage(msg));

      return { conversation, messages: chatMessages };
    } catch (error: any) {
      throw new Error(`Failed to load conversation: ${error.message}`);
    }
  }
}

export const databaseService = new DatabaseService();
export default databaseService;
