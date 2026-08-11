import re
import hashlib
import logging
import google.generativeai as genai
from..core.config import Settings
from .rate_limiter import GlobalRateLimiter

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """
Bạn là trợ lý giải thích kết quả của một mô hình phát hiện văn bản AI tiếng Việt.
Model đã dự đoán: {predicted_class} (xác suất {pred_prob:.1%}).

Bằng chứng đã tính được (KHÔNG được suy đoán thêm ngoài các số liệu này):
- Từ ảnh hưởng mạnh nhất ủng hộ AI: {top_ai_words}
- Từ ảnh hưởng mạnh nhất ủng hộ Human: {top_human_words}
- Số câu: {sentence_count}, độ dài câu trung bình: {avg_sentence_length:.1f} từ, độ lệch chuẩn: {sentence_length_std:.1f}
- Mật độ dấu câu: {punctuation_density:.3f}

Trả lời CHÍNH XÁC theo định dạng dưới đây, MỖI BULLET TRÊN MỘT DÒNG MỚI:
- Giải thích đầu tiên
- Giải thích thứ hai
- Giải thích thứ ba
(và cứ như vậy, tối đa 5 bullet)

Giải thích vì sao văn bản có đặc điểm trên, CHỈ dựa trên các số liệu đã cho, không thêm thông tin ngoài dữ liệu này.
"""

class GeminiExplanationService:
    def __init__(self, settings: Settings):
        self.enabled = bool(settings.gemini_api_key)
        self._cache: dict[str, list[str]] = {}
        self._cache_order: list[str] = []
        self._cache_max_size = 200
        self.rate_limiter = GlobalRateLimiter(max_calls=settings.gemini_rate_limit_per_minute)

        if self.enabled:
            genai.configure(api_key=settings.gemini_api_key)
            self.model_client = genai.GenerativeModel(settings.gemini_model)

    @staticmethod
    def _cache_key(text):
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def _parse_bullets(self, text: str):
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        bullet_lines = [l for l in lines if re.match(r'^[-*•]\s+', l)]

        if len(bullet_lines) >= 2:
            raw_bullets  = bullet_lines
        else:
            raw_bullets = re.split(r'(?:^|\s)[-*]\s+', text)

        bullets = []
        for b in raw_bullets:
            b = re.sub(r'^[-*•]\s*', '', b)
            b = re.sub(r'\*\*(.+?)\*\**', r'\1', b)
            b = b.strip()
            if b:
                bullets.append(b)
        return bullets if bullets else [text.strip()]

    def explain(self, cleaned_text, predicted_class, pred_prob, signals):
        if not self.enabled:
            raise RuntimeError('GEMINI_API_KEY chưa được cấu hình')

        key = self._cache_key(cleaned_text)
        if key in self._cache:
            return {
                'bullets': self._cache[key],
                'cached': True
            }

        if not self.rate_limiter.allow():
            raise RuntimeError('Đã vượt quá giới hạn gọi Gemini trong phút này, thử lại sau')

        prompt = PROMPT_TEMPLATE.format(
            predicted_class=predicted_class,
            pred_prob=pred_prob,
            top_ai_words=', '.join(signals['top_ai_words']) or '(không có)',
            top_human_words=', '.join(signals['top_human_words']) or '(không có)',
            sentence_count=signals['sentence_count'],
            avg_sentence_length=signals['avg_sentence_length'],
            sentence_length_std=signals['sentence_length_std'],
            punctuation_density=signals['punctuation_density']
        )

        try: 
            response = self.model_client.generate_content(prompt)
            text = response.text.strip()
            bullets = self._parse_bullets(text)
        except Exception as e:
            logger.exception('Gemini call failed')
            logger.error(f'Lỗi Gemini API: {e}')
            raise RuntimeError('Không gọi được Gemini API')

        self._cache[key] = bullets
        self._cache_order.append(key)
        if len(self._cache_order) > self._cache_max_size:
            self._cache.pop(self._cache_order.pop(0), None)

        return {
            'bullets': bullets,
            'cached': False
        }