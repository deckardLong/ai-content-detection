import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from .core.config import settings
from .services.model_service import ModelService
from .services.gemini_service import GeminiExplanationService
from .routers import health, explain, predict, explain_llm, auth, history
from .core.database import Base, engine

logging.basicConfig(level=logging.INFO)
os.makedirs(settings.avatar_upload_dir, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    os.makedirs(settings.avatar_upload_dir, exist_ok=True)

    service = ModelService(settings)
    service.load() # load model 1 time when server starts
    app.state.model_service = service
    app.state.gemini_service = GeminiExplanationService(settings)
    yield

    # No cleanup here
app = FastAPI(title='Vietnamese AI IT News Detector', lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=['*'],
    allow_headers=['*']
)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(explain.router)
app.include_router(explain_llm.router)
app.include_router(auth.router)
app.include_router(history.router)

app.mount('/uploads/avatars', StaticFiles(directory=settings.avatar_upload_dir), name='avatars')