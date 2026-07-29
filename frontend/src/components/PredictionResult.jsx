export default function PredictionResult({prediction}) {
    const {predicted_class, prob_human, prob_ai} = prediction;
    const aiPercent = (prob_ai * 100).toFixed(1);
    const humanPercent = (prob_human * 100).toFixed(1);

    return (
        <div className="prediction-result">
            <h3>
                Kết quả: {' '}
                <span className={predicted_class == 'AI' ? 'label-ai' : 'label-human'}>
                    {predicted_class == 'AI' ? 'AI tạo ra' : 'Con người viết'}
                </span>
            </h3>
            <div className="prob-bar">
                <div className="prob-bar-ai" style={{width: `${aiPercent}%`}} />
            </div>
            <div className="prob-labels">
                <span>AI: {aiPercent}%</span>
                <span>Human: {humanPercent}%</span>
            </div>
        </div>
    );
}