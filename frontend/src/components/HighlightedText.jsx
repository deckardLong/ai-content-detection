export default function HighlightedText({tokens, scores}) {
    const n = tokens.length;
    const indices = [...Array(n).keys()].sort(
        (a, b) => Math.abs(scores[b]) - Math.abs(scores[a])
    );
    const rank = new Array(n);
    indices.forEach((tokenIdx, order) => {
        rank[tokenIdx] = order;
    });

    const HIGHLIGHT_FRACTION = 0.35; 
    const highlightCount = Math.max(1, Math.round(n * HIGHLIGHT_FRACTION));

    return (
        <p className="highlighted-text">
        {tokens.map((token, i) => {
            const score = scores[i];
            const isHighlighted = rank[i] < highlightCount;
            let style = {};

            if (isHighlighted) {
            const strength = 1 - rank[i] / highlightCount; 
            const opacity = (0.25 + 0.55 * strength).toFixed(2);
            style.backgroundColor = score > 0
                ? `rgba(220, 53, 69, ${opacity})`  
                : `rgba(25, 135, 84, ${opacity})`; 
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