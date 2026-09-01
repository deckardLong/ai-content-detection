from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Text, JSON
from sqlalchemy.sql import func
from ..core.database import Base
from ..core.ids import generate_meaningful_id

class PredictionHistory(Base):
    __tablename__ = 'prediction_history'

    id = Column(String(40), primary_key=True, default=lambda: generate_meaningful_id('pred'))
    user_id = Column(String(40), ForeignKey('users.id'), nullable=False, index=True)
    text = Column(Text, nullable=False)
    predicted_class = Column(String(20), nullable=False)
    prob_ai = Column(Float, nullable=False)
    explain_result = Column(JSON, nullable=True)
    llm_result = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())