import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..dependencies import get_model_service, get_current_user_optional
from ..core.database import get_db
from ..services.model_service import ModelService
from ..schemas.text import TextRequest, PredictResponse
from ..models.prediction_history import PredictionHistory
from ..models.user import User

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post('/predict', response_model=PredictResponse)
def predict(req: TextRequest, service: ModelService = Depends(get_model_service), current_user: User | None = Depends(get_current_user_optional), db: Session = Depends(get_db)):
    try:
        result = service.predict(req.text)
    except Exception:
        logger.exception('Predict failed')
        raise HTTPException(status_code=500, detail='Không thể xử lý văn bản này')

    history_id = None
    print("CURRENT USER:", current_user)
    if current_user is not None:
        print("ĐANG LƯU HISTORY CHO:", current_user.id)
        entry = PredictionHistory(
            user_id=current_user.id,
            text=req.text,
            predicted_class=result['predicted_class'],
            prob_ai=result['prob_ai']
        )
        db.add(entry)
        db.commit()
        db.refresh(entry)
        print("ĐÃ LƯU:", entry.id)
        history_id = entry.id
    
    return PredictResponse(
        predicted_class=result['predicted_class'],
        prob_human=result['prob_human'],
        prob_ai=result['prob_ai'],
        history_id=history_id
    )