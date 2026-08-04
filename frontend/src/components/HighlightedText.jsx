const TIERS = [
  { upTo: 0.10, opacity: 0.75 }, // top 10% mạnh nhất
  { upTo: 0.30, opacity: 0.50 }, // 10% - 30%
  { upTo: 0.60, opacity: 0.28 }, // 30% - 60%
  { upTo: 1.00, opacity: 0.12 }, // 60% - 100% (vẫn có màu, nhưng rất nhạt)
];

function tierOpacity(rankFraction) {
  const tier = TIERS.find((t) => rankFraction <= t.upTo);
  return tier.opacity;
}

function rankFractions(items) {
  const map = new Map();
  const n = items.length;
  items.forEach((idx, order) => {
    map.set(idx, n > 1 ? order / (n - 1) : 0);
  });
  return map;
}

export default function HighlightedText({ tokens, scores }) {
  const positiveIdx = [...Array(tokens.length).keys()]
    .filter((i) => scores[i] > 0)
    .sort((a, b) => scores[b] - scores[a]);
  const negativeIdx = [...Array(tokens.length).keys()]
    .filter((i) => scores[i] < 0)
    .sort((a, b) => Math.abs(scores[b]) - Math.abs(scores[a]));

  const positiveRank = rankFractions(positiveIdx);
  const negativeRank = rankFractions(negativeIdx);

  return (
    <p className="highlighted-text">
      {tokens.map((token, i) => {
        const score = scores[i];
        let style = {};

        if (score > 0) {
          const opacity = tierOpacity(positiveRank.get(i));
          style.backgroundColor = `rgba(220, 53, 69, ${opacity})`;
        } else if (score < 0) {
          const opacity = tierOpacity(negativeRank.get(i));
          style.backgroundColor = `rgba(25, 135, 84, ${opacity})`;
        }

        return (
          <span key={i} className="token" style={style} title={`score: ${score.toFixed(3)}`}>
            {token}{' '}
          </span>
        );
      })}
    </p>
  );
}