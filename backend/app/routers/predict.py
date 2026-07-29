import logging
from fastapi import APIRouter, Depends, HTTPException
from ..dependencies import get_model_service
from ..services.model_service import ModelService
from ..schemas.text import TextRequest, PredictResponse

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post('/predict', response_model=PredictResponse)
def predict(req: TextRequest, service: ModelService = Depends(get_model_service)):
    try:
        result = service.predict(req.text)
    except Exception:
        logger.exception('Predict failed')
        raise HTTPException(status_code=500, detail='Không thể xử lý văn bản này')
    return PredictResponse(
        predicted_class=result['predicted_class'],
        prob_human=result['prob_human'],
        prob_ai=result['prob_ai']
    )