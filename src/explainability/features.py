import re
import statistics

def compute_signals(cleaned_text, tokens, scores, top_k = 8):
    paired = sorted(zip(tokens, scores), key=lambda x: abs(x[1]), reverse=True)
    top_ai_words = [w for w, s in paired if s > 0][:top_k]
    top_human_words = [w for w, s in paired if s < 0][:top_k]

    sentences = [s.strip() for s in re.split(r'[.!?]', cleaned_text) if s.strip()]
    sentence_lengths = [len(s.split()) for s in sentences]
    punct_count = len(re.findall(r'[^\w\sÀ-ỹ]', cleaned_text))

    return {
        'top_ai_words': top_ai_words,
        'top_human_words': top_human_words,
        'sentence_count': len(sentences),
        'avg_sentence_length': sum(sentence_lengths) / max(len(sentence_lengths), 1),
        'sentence_length_std': statistics.pstdev(sentence_lengths) if len(sentence_lengths) > 1 else 0.0,
        'punctuation_density': punct_count / max(len(cleaned_text), 1)
    }