export default function PredictionResult({ prediction }) {
  const { predicted_class, prob_ai } = prediction;
  const prob_human = 1 - prob_ai; 
  const aiPercent = prob_ai * 100;
  const isAi = predicted_class === "AI";

  const formatPercent = (p) => {
    const val = p * 100;
    return val.toFixed(1);
  };

  const aiDisplay = formatPercent(prob_ai);
  const humanDisplay = formatPercent(prob_human);

  return (
    <div className="panel">
      <p className="section-label">Kết quả đo</p>

      <div className="verdict-heading">
        <span className="label">Nhận định</span>
        <span className={`value ${isAi ? "is-ai" : "is-human"}`}>
          {isAi ? "AI tạo ra" : "Con người viết"}
        </span>
      </div>

      <div className="gauge">
        <div className="gauge-marker" style={{ left: `${aiPercent}%` }} />
      </div>

      <div className="gauge-readout">
        <span className="tag">
          HUMAN <span className="pct">{humanDisplay}%</span>
        </span>
        <span className="tag">
          AI <span className="pct">{aiDisplay}%</span>
        </span>
      </div>
    </div>
  );
}