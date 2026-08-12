import { createContext, useContext, useEffect, useState } from 'react';
import { fetchCurrentUser, loginUser, registerUser } from '../api';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const token = localStorage.getItem('access_token');
    if (!token) {
      setLoading(false);
      return;
    }
    fetchCurrentUser()
      .then(setUser)
      .catch(() => localStorage.removeItem('access_token'))
      .finally(() => setLoading(false));
  }, []);

  async function login(username, password) {
    const data = await loginUser(username, password);
    localStorage.setItem('access_token', data.access_token);
    setUser(data.user);
  }

  async function register(username, password) {
    await registerUser(username, password);
  }

  function logout() {
    localStorage.removeItem('access_token');
    setUser(null);
  }

  function updateUser(updatedUser) {
    setUser(updatedUser);
  }

  return (
    <AuthContext.Provider value={{ user, loading, login, register, logout, updateUser }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}