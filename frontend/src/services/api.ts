import axios from 'axios';

// Unified backend on port 8000
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds
  headers: {
    'Content-Type': 'application/json',
  },
});

// Legacy exports for backward compatibility
const authApi = api;
const aiApi = api;

// Request interceptor
api.interceptors.request.use(
  (config) => {
    console.log(`Making ${config.method?.toUpperCase()} request to ${config.url}`);
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface ChatRequest {
  message: string;
  settings: {
    tokens: number;
    temperature: number;
    topP: number;
    repetitionPenalty: number;
    model: string;
  };
  enhance: boolean;
  history?: ChatMessage[];
}

export interface ChatResponse {
  response: string;
  enhanced: boolean;
  system_message?: string;
  tokens_used?: number;
  processing_time?: number;
}

export interface CompletionRequest {
  prompt: string;
  settings: {
    tokens: number;
    temperature: number;
    topP: number;
    repetitionPenalty: number;
    model: string;
  };
  enhance: boolean;
}

export interface CompletionResponse {
  completion: string;
  enhanced: boolean;
  tokens_used?: number;
  processing_time?: number;
}

export interface StatusResponse {
  connected: boolean;
  loaded: boolean;
  model_name?: string;
  lm_studio_available?: boolean;
}

// API Methods
export const chatAPI = {
  sendMessage: async (request: ChatRequest): Promise<ChatResponse> => {
    const response = await aiApi.post('/api/chat', request);
    return response.data;
  },

  getStatus: async (): Promise<StatusResponse> => {
    const response = await aiApi.get('/api/status');
    return response.data;
  },
};

export const completionAPI = {
  generate: async (request: CompletionRequest): Promise<CompletionResponse> => {
    const response = await aiApi.post('/api/completion', request);
    return response.data;
  },
};

export const lmStudioAPI = {
  getStatus: async (): Promise<{ connected: boolean }> => {
    const response = await aiApi.get('/api/lm-studio/status');
    return response.data;
  },

  reconnect: async (): Promise<{ success: boolean; message: string }> => {
    const response = await aiApi.post('/api/lm-studio/reconnect');
    return response.data;
  },

  setInstruction: async (instruction: string): Promise<{ success: boolean }> => {
    const response = await aiApi.post('/api/lm-studio/instruction', { instruction });
    return response.data;
  },
};

export const modelAPI = {
  getStatus: async (): Promise<{ loaded: boolean; model_name?: string }> => {
    const response = await aiApi.get('/api/model/status');
    return response.data;
  },

  loadModel: async (modelName: string): Promise<{ success: boolean; message: string }> => {
    const response = await aiApi.post('/api/model/load', { model_name: modelName });
    return response.data;
  },
};

export { authApi, aiApi, api };
export default api;
