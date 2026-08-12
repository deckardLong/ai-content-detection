import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..dependencies import get_model_service, get_current_user_optional
from ..core.database import get_db
from ..services.model_service import ModelService
from ..schemas.text import TextRequest, ExplainResponse
from ..models.prediction_history import PredictionHistory
from ..models.user import User

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post('/explain', response_model=ExplainResponse)
def explain(req: TextRequest, service: ModelService = Depends(get_model_service), current_user: User | None = Depends(get_current_user_optional), db: Session = Depends(get_db)):
    try:
        result = service.explain(req.text)
    except Exception:
        logger.exception('Explain failed')
        raise HTTPException(status_code=500, detail='Không thể giải thích văn bản này')

    if current_user is not None and req.history_id:
        entry = (
            db.query(PredictionHistory)
            .filter(PredictionHistory.id == req.history_id, PredictionHistory.user_id == current_user.id)
            .first()
        )
        if entry is not None:
            entry.explain_result = {
                'tokens': list(result['tokens']),
                'scores': [float(s) for s in result['scores']]
            }
            db.commit()

    return ExplainResponse(
        tokens=result['tokens'],
        scores=result['scores'],
        predicted_label=result['predicted_label'],
        pred_prob=result['pred_prob'],
    )