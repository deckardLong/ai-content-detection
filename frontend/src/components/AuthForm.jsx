import { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';

export default function AuthForm() {
  const { login, register } = useAuth();
  const [mode, setMode] = useState('login');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState(null);
  const [successMessage, setSuccessMessage] = useState(null);
  const [loading, setLoading] = useState(false);

  function resetFields() {
    setPassword('');
    setConfirmPassword('');
  }

  function switchMode(nextMode) {
    setMode(nextMode);
    setError(null);
    setSuccessMessage(null);
    resetFields();
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setError(null);
    setSuccessMessage(null);

    if (mode === 'register' && password !== confirmPassword) {
      setError('Mật khẩu nhập lại không khớp.');
      return;
    }

    setLoading(true);
    try {
      if (mode === 'login') {
        await login(username, password);
      } else {
        await register(username, password);
        setSuccessMessage('Đăng ký thành công! Hãy đăng nhập để tiếp tục.');
        switchMode('login');
        setUsername(username); // keep username
      }
    } catch (err) {
      setError(err.message || 'Có lỗi xảy ra, thử lại sau.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="auth-shell">
      <div className="auth-card panel">
        <p className="section-label">{mode === 'login' ? 'Đăng nhập' : 'Tạo tài khoản'}</p>
        <form onSubmit={handleSubmit} className="auth-form">
          <label>
            Tên đăng nhập
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="ví dụ: minhanh"
              required
              minLength={3}
            />
          </label>
          <label>
            Mật khẩu
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="ít nhất 6 ký tự"
              required
              minLength={6}
            />
          </label>
          {mode === 'register' && (
            <label>
              Nhập lại mật khẩu
              <input
                type="password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                placeholder="nhập lại mật khẩu ở trên"
                required
                minLength={6}
              />
            </label>
          )}
          {error && <p className="error-banner">{error}</p>}
          {successMessage && <p className="success-banner">{successMessage}</p>}
          <button className="btn-primary" type="submit" disabled={loading}>
            {loading ? 'Đang xử lý...' : mode === 'login' ? 'Đăng nhập' : 'Đăng ký'}
          </button>
        </form>
        <p className="auth-switch">
          {mode === 'login' ? 'Chưa có tài khoản?' : 'Đã có tài khoản?'}{' '}
          <button type="button" className="link-btn" onClick={() => switchMode(mode === 'login' ? 'register' : 'login')}>
            {mode === 'login' ? 'Đăng ký ngay' : 'Đăng nhập'}
          </button>
        </p>
      </div>
    </div>
  );
}