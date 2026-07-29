import { useState } from 'react';
import { predictText, explainText } from './api';
import PredictionResult from './components/PredictionResult';
import HighlightedText from './components/HighlightedText';

export default function App() {
  const [text, setText] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [loadingPredict, setLoadingPredict] = useState(false);
  const [loadingExplain, setLoadingExplain] = useState(false);
  const [error, setError] = useState(null);

  async function handlePredict() {
    if (!text.trim()) return;
    setError(null);
    setPrediction(null);
    setExplanation(null);
    setLoadingPredict(true);
    try {
      setPrediction(await predictText(text));
    } catch (err) {
      console.error('Predict error:', err);
      setError('Không thể phân tích văn bản. Kiểm tra server đã chạy chưa.');
    } finally {
      setLoadingPredict(false);
    }
  }

  async function handleExplain() {
    setError(null);
    setLoadingExplain(true);
    try {
      setExplanation(await explainText(text));
    } catch {
      setError('Không thể tạo giải thích cho văn bản này.');
    } finally {
      setLoadingExplain(false);
    }
  }

  return (
    <div className="app-container">
      <h1>Vietnamese AI IT News Detector</h1>
      <textarea
        rows={8}
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Dán văn bản tiếng Việt cần kiểm tra vào đây..."
      />
      <div className="actions">
        <button onClick={handlePredict} disabled={loadingPredict || !text.trim()}>
          {loadingPredict ? 'Đang phân tích...' : 'Phân tích'}
        </button>
        {prediction && (
          <button onClick={handleExplain} disabled={loadingExplain}>
            {loadingExplain ? 'Đang tạo giải thích...' : 'Xem giải thích'}
          </button>
        )}
      </div>

      {error && <p className="error">{error}</p>}
      {prediction && <PredictionResult prediction={prediction} />}
      {explanation && (
        <div className="explanation">
          <h3>Giải thích (từ nào ảnh hưởng tới kết quả)</h3>
          <HighlightedText tokens={explanation.tokens} scores={explanation.scores} />
        </div>
      )}
    </div>
  );
}