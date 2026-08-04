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
      setError('Không thể phân tích văn bản. Kiểm tra backend đã chạy chưa.');
    } finally {
      setLoadingPredict(false);
    }
  }

  async function handleExplain() {
    setError(null);
    setLoadingExplain(true);
    try {
      setExplanation(await explainText(text));
    } catch (err) {
      console.error('Explain error:', err);
      setError('Không thể tạo giải thích cho văn bản này.');
    } finally {
      setLoadingExplain(false);
    }
  }

  return (
    <div className="app-shell">
      <div className="device-strip">
        <span className="dot" />
        <span>VĂN.DETECT</span>
        <span className="spacer" />
        <span>model: bamibert</span>
        <span>·</span>
        <span>device: cpu</span>
      </div>

      <div className="masthead">
        <p className="eyebrow">Phân tích bài báo IT tiếng Việt</p>
        <h1>Vietnamese AI IT News Detector</h1>
        <p>Dán văn bản vào bên dưới để đo khả năng được tạo bởi AI.</p>
      </div>

      <div className="panel">
        <p className="section-label">Văn bản đầu vào</p>
        <textarea
          rows={8}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Dán văn bản tiếng Việt cần kiểm tra vào đây..."
        />
        <div className="actions">
          <button className="btn-primary" onClick={handlePredict} disabled={loadingPredict || !text.trim()}>
            {loadingPredict ? 'Đang phân tích...' : 'Phân tích'}
          </button>
          {prediction && (
            <button className="btn-secondary" onClick={handleExplain} disabled={loadingExplain}>
              {loadingExplain ? 'Đang tạo giải thích...' : 'Xem giải thích'}
            </button>
          )}
        </div>
        {error && <p className="error-banner">{error}</p>}
      </div>

      {prediction && <PredictionResult prediction={prediction} />}

      {explanation && (
        <div className="panel">
          <p className="section-label">Bằng chứng phân tích</p>
          <HighlightedText tokens={explanation.tokens} scores={explanation.scores} />
          <div className="legend">
            <span><span className="swatch" style={{ background: 'rgba(224,41,58,0.55)' }} />Ủng hộ AI</span>
            <span><span className="swatch" style={{ background: 'rgba(21,150,82,0.55)' }} />Ủng hộ con người</span>
          </div>
        </div>
      )}
    </div>
  );
}