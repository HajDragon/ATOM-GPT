import { authApi } from './api';
import axios from 'axios';

const API_BASE_URL = ''; // Using authApi baseURL

export interface User {
  id: number;
  username: string;
  email: string;
  first_name?: string;
  last_name?: string;
  is_active: boolean;
  is_admin: boolean;
  created_at: string;
  updated_at: string;
}

export interface UserStats {
  total_conversations: number;
  chat_conversations: number;
  completion_conversations: number;
  total_messages: number;
  total_requests: number;
  total_tokens: number;
  avg_response_time: number;
  enhanced_requests: number;
  user_id: number;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  username: string;
  email: string;
  password: string;
  first_name?: string;
  last_name?: string;
}

export interface AuthResponse {
  access_token: string;
  user: User;
  message: string;
}

class AuthService {
  private token: string | null = null;
  private user: User | null = null;

  constructor() {
    // Load token from localStorage on initialization
    this.token = localStorage.getItem('auth_token');
    const userData = localStorage.getItem('user_data');
    if (userData) {
      try {
        this.user = JSON.parse(userData);
      } catch (error) {
        console.error('Failed to parse user data from localStorage:', error);
        localStorage.removeItem('user_data');
      }
    }
    
    // Set auth header if we have a token
    this.setAuthHeader();
  }

  // Set authentication header for axios
  private setAuthHeader() {
    if (this.token) {
      console.log('🔐 Setting auth header with token:', this.token.substring(0, 20) + '...');
      authApi.defaults.headers.common['Authorization'] = `Bearer ${this.token}`;
    } else {
      console.log('🔐 No token available, removing auth header');
      delete authApi.defaults.headers.common['Authorization'];
    }
  }

  async login(credentials: LoginRequest): Promise<AuthResponse> {
    try {
      console.log('🔐 Attempting login with:', { email: credentials.email });
      console.log('🔗 Auth API base URL:', authApi.defaults.baseURL);
      console.log('🔗 Full URL will be:', `${authApi.defaults.baseURL}/../auth/login`);
      
      // Auth endpoints are at root level, not under /api
      const response = await axios.post('http://localhost:8000/auth/login', credentials);
      console.log('✅ Login response:', response.status, response.data);
      
      const { access_token, user, message } = response.data;

      this.token = access_token;
      this.user = user;

      // Store in localStorage
      localStorage.setItem('auth_token', access_token);
      localStorage.setItem('user_data', JSON.stringify(user));

      this.setAuthHeader();

      return { access_token, user, message };
    } catch (error: any) {
      console.error('❌ Login error:', error);
      console.error('Error response:', error.response?.data);
      console.error('Error status:', error.response?.status);
      console.error('Error message:', error.message);
      throw new Error(error.response?.data?.error || 'Login failed');
    }
  }

  async register(userData: RegisterRequest): Promise<AuthResponse> {
    try {
      const response = await axios.post('http://localhost:8000/auth/register', userData);
      const { access_token, user, message } = response.data;

      this.token = access_token;
      this.user = user;

      // Store in localStorage
      localStorage.setItem('auth_token', access_token);
      localStorage.setItem('user_data', JSON.stringify(user));

      this.setAuthHeader();

      return { access_token, user, message };
    } catch (error: any) {
      throw new Error(error.response?.data?.error || 'Registration failed');
    }
  }

  async getCurrentUser(): Promise<{ user: User; stats: UserStats }> {
    try {
      this.setAuthHeader();
      const response = await axios.get('http://localhost:8000/auth/me', {
        headers: { Authorization: `Bearer ${this.token}` }
      });
      this.user = response.data.user;
      localStorage.setItem('user_data', JSON.stringify(this.user));
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.error || 'Failed to get user info');
    }
  }

  async getUserSettings(): Promise<Record<string, any>> {
    try {
      // TODO: Implement settings endpoint in backend
      // For now, return empty settings from localStorage
      const settings = localStorage.getItem('user_settings');
      return settings ? JSON.parse(settings) : {};
    } catch (error: any) {
      console.warn('Settings not available, using local storage');
      return {};
    }
  }

  async updateUserSettings(settings: Record<string, any>): Promise<Record<string, any>> {
    try {
      // TODO: Implement settings endpoint in backend
      // For now, save to localStorage
      localStorage.setItem('user_settings', JSON.stringify(settings));
      return settings;
    } catch (error: any) {
      throw new Error('Failed to update settings');
    }
  }

  logout() {
    console.log('🚪 Logging out user...');
    this.token = null;
    this.user = null;
    localStorage.removeItem('auth_token');
    localStorage.removeItem('user_data');
    localStorage.removeItem('user_settings');
    // Clear any cached conversations or other user data
    localStorage.removeItem('conversations_cache');
    delete authApi.defaults.headers.common['Authorization'];
    console.log('✅ User logged out and data cleared');
  }

  isAuthenticated(): boolean {
    return !!this.token;
  }

  getUser(): User | null {
    return this.user;
  }

  getToken(): string | null {
    return this.token;
  }

  // Initialize auth headers on app startup
  initialize() {
    console.log('🔄 Initializing auth service...');
    this.setAuthHeader();
  }
}

export const authService = new AuthService();
export default authService;
