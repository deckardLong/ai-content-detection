from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_name: str = 'Qualcomm-AI-Research/BamiBERT' 
    checkpoint_path: str = 'models/best_model.pt'
    max_length: int = 512
    device: str = 'cpu'
    ig_n_steps: int = 25
    cors_origins: list[str] = ['http://localhost:5173', 'http://localhost:3000']

    gemini_api_key: str = ''
    gemini_model: str = 'gemini-2.5-flash'
    gemini_rate_limit_per_minute: int = 10

    # Auth
    jwt_secret_key: str = ''
    jwt_algorithm: str = 'HS256'
    jwt_expire_minutes: int = 1440
    database_url: str = 'sqlite:///./database/app.db'
    avatar_upload_dir: str = 'uploads/avatars'

    class Config:
        env_file = '.env'
        env_prefix = 'app_'

settings = Settings()