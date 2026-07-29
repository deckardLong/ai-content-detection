import logging
from fastapi import APIRouter, Depends, HTTPException
from ..dependencies import get_model_service
from ..services.model_service import ModelService
from ..schemas.text import TextRequest, ExplainResponse

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post('/explain', response_model=ExplainResponse)
def explain(req: TextRequest, service: ModelService = Depends(get_model_service)):
    try:
        pred = service.predict(req.text)
        result = service.explain(pred['cleaned_text'], target_label=pred['predicted_label'])
    except Exception:
        logger.exception('Explain failed')
        raise HTTPException(status_code=500, detail='Không thể giải thích văn bản này')
    return ExplainResponse(**result)