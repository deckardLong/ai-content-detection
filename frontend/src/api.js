const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

function authHeaders() {
  const token = localStorage.getItem('access_token');
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function postJSON(path, body, extraHeaders = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...extraHeaders },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `${path} failed with status ${res.status}`);
  }
  return res.json();
}

async function getJSON(path, extraHeaders = {}) {
  const res = await fetch(`${API_BASE}${path}`, { headers: extraHeaders });
  if (!res.ok) throw new Error(`${path} failed with status ${res.status}`);
  return res.json();
}

export const predictText = (text) => postJSON('/predict', { text });
export const explainText = (text) => postJSON('/explain', { text });
export const explainLlmText = (text) => postJSON('/explain-llm', { text });

export const registerUser = (username, password) => postJSON('/auth/register', { username, password });
export const loginUser = (username, password) => postJSON('/auth/login', { username, password });
export const fetchCurrentUser = () => getJSON('/auth/me', authHeaders());

export async function uploadAvatar(file) {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE}/auth/avatar`, {
    method: 'POST',
    headers: authHeaders(), 
    body: formData,
  });
  if (!res.ok) throw new Error('Tải avatar thất bại');
  return res.json();
}

export const AVATAR_BASE = API_BASE;