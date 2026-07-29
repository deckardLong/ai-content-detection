from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_name: str = 'Qualcomm-AI-Research/BamiBERT' 
    checkpoint_path: str = 'models/best_model.pt'
    max_length: int = 512
    device: str = 'cpu'
    ig_n_steps: int = 25
    cors_origins: list[str] = ['http://localhost:5173', 'http://localhost:3000']

    class Config:
        env_file = '.env'
        env_prefix = 'APP_'

settings = Settings()