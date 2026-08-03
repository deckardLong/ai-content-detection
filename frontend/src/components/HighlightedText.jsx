export default function HighlightedText({tokens, scores}) {
    const n = tokens.length;
    const indices = [...Array(n).keys()].sort(
        (a, b) => Math.abs(scores[b]) - Math.abs(scores[a])
    );
    const rank = new Array(n);
    indices.forEach((tokenIdx, order) => {
        rank[tokenIdx] = order;
    });

    const MIN_OPACITY = 0.08; 
    const MAX_OPACITY = 0.75; 
    const POWER = 2;          

    return (
        <p className="highlighted-text">
        {tokens.map((token, i) => {
            const score = scores[i];
            const strength = n > 1 ? 1 - rank[i] / (n - 1) : 1;
            const opacity = (MIN_OPACITY + (MAX_OPACITY - MIN_OPACITY) * strength ** POWER).toFixed(2);

            const backgroundColor = score > 0
                ? `rgba(220, 53, 69, ${opacity})`  // đỏ: đẩy về phía nhãn dự đoán (AI)
                : `rgba(25, 135, 84, ${opacity})`; // xanh: đẩy ngược lại (Human)

            return (
                <span
                    key={i}
                    className="token"
                    style={{ backgroundColor }}
                    title={`score: ${score.toFixed(3)}`}
                >
                    {token}{' '}
                </span>
            );
        })}
        </p>
    );
}