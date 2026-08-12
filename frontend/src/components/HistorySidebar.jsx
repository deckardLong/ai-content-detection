import { useEffect, useState } from 'react';
import { fetchHistory, fetchHistoryItem, deleteHistoryItem } from '../api';

export default function HistorySidebar({ onSelect, refreshKey }) {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeId, setActiveId] = useState(null);

  useEffect(() => {
    loadHistory();
  }, [refreshKey]);

  async function loadHistory() {
    setLoading(true);
    try {
      setItems(await fetchHistory());
    } catch (err) {
      console.error('Load history error:', err);
    } finally {
      setLoading(false);
    }
  }

  async function handleSelect(id) {
    setActiveId(id);
    try {
      onSelect(await fetchHistoryItem(id));
    } catch (err) {
      console.error('Load history detail error:', err);
    }
  }

  async function handleDelete(e, id) {
    e.stopPropagation();
    try {
      await deleteHistoryItem(id);
      setItems((prev) => prev.filter((it) => it.id !== id));
    } catch (err) {
      console.error('Delete history error:', err);
    }
  }

  function formatTime(iso) {
    return new Date(iso).toLocaleString('vi-VN', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' });
  }

  return (
    <aside className="history-sidebar">
      <p className="sidebar-title">Lịch sử phân tích</p>
      {loading && <p className="sidebar-hint">Đang tải...</p>}
      {!loading && items.length === 0 && <p className="sidebar-hint">Chưa có lịch sử nào.</p>}
      <ul className="history-list">
        {items.map((item) => (
          <li
            key={item.id}
            className={`history-item ${activeId === item.id ? 'active' : ''}`}
            onClick={() => handleSelect(item.id)}
          >
            <div className="history-item-main">
              <span className={`history-badge ${item.predicted_class === 'AI' ? 'badge-ai' : 'badge-human'}`}>
                {item.predicted_class}
              </span>
              <span className="history-text">{item.text_preview}</span>
            </div>
            {(item.has_explain || item.has_llm) && (
              <div className="history-item-tags">
                {item.has_explain && <span className="tag-icon" title="Có highlight">🖍</span>}
                {item.has_llm && <span className="tag-icon" title="Có giải thích AI">✨</span>}
              </div>
            )}
            <div className="history-item-footer">
              <span className="history-time">{formatTime(item.created_at)}</span>
              <button type="button" className="history-delete" onClick={(e) => handleDelete(e, item.id)} title="Xóa">
                ×
              </button>
            </div>
          </li>
        ))}
      </ul>
    </aside>
  );
}