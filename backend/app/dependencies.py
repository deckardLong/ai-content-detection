from fastapi import Request

def get_model_service(request: Request):
    return request.app.state.model_service