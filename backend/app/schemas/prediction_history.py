from pydantic import BaseModel
from datetime import datetime

class PredictionHistoryOut(BaseModel):
    id: str
    text_preview: str
    predicted_class: str
    prob_ai: float
    created_at: datetime
    has_explain: bool = False
    has_llm: bool = False

class PredictionHistoryDetail(BaseModel):
    id: str
    text: str
    predicted_class: str
    prob_ai: float
    created_at: datetime  
    explain_result: dict | None = None
    llm_result: dict | None = None