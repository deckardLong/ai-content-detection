from fastapi import Request

def get_model_service(request: Request):
    return request.app.state.model_service

def get_gemini_service(request: Request):
    return request.app.state.gemini_service