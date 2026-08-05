export default function LlmExplanation({bullets, cached}) {
  return (
    <div>
      <ul className="llm-bullets">
        {bullets.map((b, i) => <li key={i}>{b}</li>)}
      </ul>
      {cached && <p className="cache-note">(kết quả từ cache)</p>}
    </div>
  );
}