from fastapi import Request, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from .core.database import get_db
from .core.security import decode_access_token
from .models.user import User

bearer_scheme = HTTPBearer() 

def get_model_service(request: Request):
    return request.app.state.model_service

def get_gemini_service(request: Request):
    return request.app.state.gemini_service

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme), db: Session = Depends(get_db)):
    user_id = decode_access_token(credentials.credentials)
    if user_id is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Token không hợp lệ hoặc đã hết hạn')
    user = db.query(User).filter(User.id == user_id).first()

    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Người dùng không tồn tại')
    return user