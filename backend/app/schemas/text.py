from pydantic import BaseModel, Field

class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000)

class PredictResponse(BaseModel):
    predicted_class: str
    prob_human: float
    prob_ai: float

class ExplainResponse(BaseModel):
    tokens: list[str]
    scores: list[float]
    predicted_label: int
    pred_prob: float