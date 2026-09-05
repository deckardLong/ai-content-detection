import re

def score_groundedness(bullets: list[str], signals: dict):
    if not signals:
        return {
            'known_words_total': 0,
            'known_words_referenced': 0,
            'numeric_claims_matched': 0,
            'punctuation_density_match': 0.0
        }

    joined = ' '.join(bullets).lower()
    known_words = set(
        w.lower() for w in signals.get('top_ai_words', []) + signals.get('top_human_words', [])
    )

    words_mentioned = sum(1 for w in known_words if w in joined)

    # Numeric claims
    numeric_claims_in_signals = {
        round(signals.get('avg_sentence_length', 0), 1),
        round(signals.get('sentence_length_std', 0), 1),
        float(signals.get('sentence_count', 0)),
    }
    numbers_in_bullets = set(float(n) for n in re.findall(r'\d+\.?\d*', joined))
    numbers_matched = len(numbers_in_bullets & numeric_claims_in_signals)

    # Punctuation density
    bullets_punctuation_density = len(re.findall(r'[^\w\sÀ-ỹ]', joined)) / max(len(joined), 1)
    signals_punctuation_density = signals.get('punctuation_density', 0)

    # If error < 10% => match
    punctuation_match = abs(bullets_punctuation_density - signals_punctuation_density) < 0.1

    return {
        'known_words_total': len(known_words),
        'known_words_referenced': words_mentioned,
        'numeric_claims_matched': numbers_matched,
        'bullets_punctuation_density': round(bullets_punctuation_density, 3),
        'signals_punctuation_density': round(signals_punctuation_density, 3),
        'punctuation_density_match': 1.0 if punctuation_match else 0.0
    }