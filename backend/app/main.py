import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.config import settings
from .services.model_service import ModelService
from .services.gemini_service import GeminiExplanationService
from .routers import health, explain, predict, explain_llm

logging.basicConfig(level=logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
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