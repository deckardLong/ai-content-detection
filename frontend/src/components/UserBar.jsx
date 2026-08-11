import { useRef, useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { uploadAvatar, AVATAR_BASE } from '../api';

export default function UserBar() {
  const { user, logout, updateUser } = useAuth();
  const fileInputRef = useRef(null);
  const [uploading, setUploading] = useState(false);

  if (!user) return null;

  async function handleFileChange(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    try {
      const updated = await uploadAvatar(file);
      updateUser(updated);
    } catch (err) {
      console.error('Upload avatar error:', err);
    } finally {
      setUploading(false);
      e.target.value = '';
    }
  }

  const initial = user.username.charAt(0).toUpperCase();

  return (
    <div className="user-bar">
      <button
        type="button"
        className="avatar-btn"
        onClick={() => fileInputRef.current?.click()}
        title="Bấm để đổi avatar"
        disabled={uploading}
      >
        {user.avatar_url ? (
          <img src={`${AVATAR_BASE}${user.avatar_url}`} alt={user.username} className="avatar-img" />
        ) : (
          <span className="avatar-placeholder">{initial}</span>
        )}
      </button>
      <input
        ref={fileInputRef}
        type="file"
        accept="image/png,image/jpeg,image/webp"
        onChange={handleFileChange}
        style={{ display: 'none' }}
      />
      <div className="user-info">
        <span className="user-name">{user.username}</span>
        <span className="user-hint">{uploading ? 'Đang tải avatar...' : 'Bấm avatar để đổi ảnh'}</span>
      </div>
      <button type="button" className="btn-secondary sign-out-btn" onClick={logout}>
        Đăng xuất
      </button>
    </div>
  );
}