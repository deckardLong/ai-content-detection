from fastapi import APIRouter, Depends
from ..dependencies import get_model_service
from ..services.model_service import ModelService

router = APIRouter()

@router.get('/health')
def health(service: ModelService = Depends(get_model_service)):
    return {
        'status': 'ok',
        'model_loaded': service.model is not None,
        'device': str(service.device)
    }