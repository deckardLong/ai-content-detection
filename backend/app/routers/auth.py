import os
import uuid
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.orm import Session

from ..core.config import settings
from ..core.database import get_db
from ..core.security import hash_password, verify_password, create_access_token
from ..dependencies import get_current_user
from ..models.user import User
from ..schemas.auth import UserCreate, UserLogin, UserOut, Token

router = APIRouter(prefix='/auth', tags=['auth'])

# User Out
def _to_user_out(user: User):
    avatar_url = f'/uploads/avatars/{user.avatar_path}' if user.avatar_path else None

    return UserOut(id=user.id, username=user.username, avatar_url=avatar_url)

# Register
@router.post('/register', response_model=UserOut)
def register(payload: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.username == payload.username).first()

    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Tên đăng nhập đã tồn tại')
    user = User(username=payload.username, hashed_password=hash_password(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)

    return _to_user_out(user)

# Login
@router.post('/login', response_model=Token)
def login(payload: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.username).first()

    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Sai tên đăng nhập hoặc mật khẩu')

    token = create_access_token(user_id=user.id)
    return Token(access_token=token, user=_to_user_out(user))

# Me
@router.get('/me', response_model=UserOut)
def me(current_user: User = Depends(get_current_user)):
    return _to_user_out(current_user)

# Post Avatar
@router.post('/avatar', response_model=UserOut)
def upload_avatar(file: UploadFile = File(...), current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    allowed_ext = {'.png', '.jpg', '.jpeg', '.webp'}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_ext:
        raise HTTPException(status_code=400, detail='Chỉ hỗ trợ ảnh PNG, JPG, JPEG, WEBP')

    os.makedirs(settings.avatar_upload_dir, exist_ok=True)
    filename = f'{current_user.id}_{uuid.uuid4().hex[:8]}{ext}'
    filepath = os.path.join(settings.avatar_upload_dir, filename)

    with open(filepath, 'wb') as f:
        f.write(file.file.read())

    current_user.avatar_path = filename
    db.commit()
    db.refresh(current_user)
    return _to_user_out(current_user)