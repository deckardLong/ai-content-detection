from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from ..core.database import Base
from ..core.ids import generate_meaningful_id

class User(Base):
    __tablename__ = 'users'

    id = Column(String(40), primary_key=True, default=lambda: generate_meaningful_id('usr'))
    username = Column(String(50), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    avatar_path = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())