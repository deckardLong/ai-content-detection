from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..dependencies import get_current_user
from ..models.user import User
from ..models.prediction_history import PredictionHistory
from ..schemas.prediction_history import PredictionHistoryOut, PredictionHistoryDetail

router = APIRouter(prefix='/predictions', tags=['history'])

@router.get('', response_model=list[PredictionHistoryOut])
def list_history(
    limit: int = Query(default=30, le=100), 
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)):
    rows = (
        db.query(PredictionHistory)
        .filter(PredictionHistory.user_id == current_user.id)
        .order_by(PredictionHistory.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        PredictionHistoryOut(
            id=r.id,
            text_preview=(r.text[:60] + '...') if len(r.text) > 60 else r.text,
            predicted_class=r.predicted_class,
            prob_ai=r.prob_ai,
            created_at=r.created_at,
            has_explain=r.explain_result is not None,
            has_llm=r.llm_result is not None
        )
        for r in rows
    ]

@router.get('/{item_id}', response_model=PredictionHistoryDetail)
def get_history_item(item_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    row = (
        db.query(PredictionHistory)
        .filter(PredictionHistory.id == item_id, PredictionHistory.user_id == current_user.id)
        .first()
    )
    if row is None:
        raise HTTPException(status_code=404, detail='Không tìm thấy')
    return PredictionHistoryDetail(
        id=row.id, 
        text=row.text, 
        predicted_class=row.predicted_class,
        prob_ai=row.prob_ai, 
        created_at=row.created_at,
        explain_result=row.explain_result,
        llm_result=row.llm_result
    )

@router.delete('/{item_id}')
def delete_history_item(item_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    row = (
        db.query(PredictionHistory)
        .filter(PredictionHistory.id == item_id, PredictionHistory.user_id == current_user.id)
        .first()
    )
    if row is None:
        raise HTTPException(status_code=404, detail='Không tìm thấy')
    db.delete(row)
    db.commit()
    return {'deleted': True}