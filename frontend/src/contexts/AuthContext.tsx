import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import authService, { User, UserStats } from '../services/auth';

interface AuthContextType {
  user: User | null;
  stats: UserStats | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (userData: {
    username: string;
    email: string;
    password: string;
    first_name?: string;
    last_name?: string;
  }) => Promise<void>;
  logout: () => void;
  refreshUserData: () => Promise<void>;
  error: string | null;
  clearError: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [stats, setStats] = useState<UserStats | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const clearError = () => setError(null);

  const refreshUserData = async () => {
    try {
      if (authService.isAuthenticated()) {
        const userData = await authService.getCurrentUser();
        setUser(userData.user);
        setStats(userData.stats);
      }
    } catch (err: any) {
      console.error('Failed to refresh user data:', err.message);
      // Don't set error here as it's a background operation
    }
  };

  useEffect(() => {
    const initializeAuth = async () => {
      setIsLoading(true);
      try {
        // Initialize auth service
        authService.initialize();
        
        // If user is logged in, get their data
        if (authService.isAuthenticated()) {
          await refreshUserData();
        } else {
          setUser(null);
          setStats(null);
        }
      } catch (err: any) {
        console.error('Auth initialization failed:', err.message);
        authService.logout();
        setUser(null);
        setStats(null);
      } finally {
        setIsLoading(false);
      }
    };

    initializeAuth();
  }, []);

  const login = async (email: string, password: string) => {
    try {
      setError(null);
      setIsLoading(true);
      
      const response = await authService.login({ email, password });
      setUser(response.user);
      
      // Get user stats
      await refreshUserData();
    } catch (err: any) {
      setError(err.message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  const register = async (userData: {
    username: string;
    email: string;
    password: string;
    first_name?: string;
    last_name?: string;
  }) => {
    try {
      setError(null);
      setIsLoading(true);
      
      const response = await authService.register(userData);
      setUser(response.user);
      
      // Get user stats
      await refreshUserData();
    } catch (err: any) {
      setError(err.message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  const logout = () => {
    authService.logout();
    setUser(null);
    setStats(null);
    setError(null);
  };

  const value: AuthContextType = {
    user,
    stats,
    isAuthenticated: !!user,
    isLoading,
    login,
    register,
    logout,
    refreshUserData,
    error,
    clearError
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};
