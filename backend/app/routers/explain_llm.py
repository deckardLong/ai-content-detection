import logging
from fastapi import APIRouter, Depends, HTTPException
from ..dependencies import get_model_service, get_gemini_service
from ..services.model_service import ModelService
from ..services.gemini_service import GeminiExplanationService
from ..schemas.text import TextRequest, ExplainLLMResponse
from src.explainability.features import compute_signals

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/explain-llm", response_model=ExplainLLMResponse)
def explain_llm(req: TextRequest,
    model_service: ModelService = Depends(get_model_service),
    gemini_service: GeminiExplanationService = Depends(get_gemini_service)
):
    try:
        pred = model_service.predict(req.text)
        attribution = model_service.explain(pred["cleaned_text"])
        signals = compute_signals(pred["cleaned_text"], attribution["tokens"], attribution["scores"])
        if pred['predicted_class'] == 'AI':
            pred_prob = pred['prob_ai']
        else:
            pred_prob = pred['prob_human']
        result = gemini_service.explain(
            pred['cleaned_text'], pred['predicted_class'], pred_prob, signals
        )
    except RuntimeError as e:
        raise HTTPException(status_code=429 if "giới hạn" in str(e) else 503, detail=str(e))
    except Exception:
        logger.exception("explain-llm failed")
        raise HTTPException(status_code=500, detail="Không thể tạo giải thích AI")
    return ExplainLLMResponse(**result, signals=signals)