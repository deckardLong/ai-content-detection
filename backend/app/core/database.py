import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from .config import settings

db_url = settings.database_url

if db_url.startswith('sqlite'):
    connect_args = {'check_same_thread': False} # allow SQLite and FastAPI to use the same thread
    if db_url.startswith("sqlite:///") and not db_url.startswith("sqlite:////"):
        raw_path = db_url.replace("sqlite:///", "")
        # Get root/ 
        base_dir = Path(__file__).resolve().parent.parent.parent
        abs_db_path = (base_dir / raw_path).resolve()
    
        abs_db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Update URL by abs URL
        db_url = f"sqlite:///{abs_db_path}"
else:
    connect_args = {}

engine = create_engine(db_url, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally: 
        db.close()