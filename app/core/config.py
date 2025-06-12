from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Video Description API"
    PROJECT_VERSION: str = "1.0.0"
    OPENAI_API_KEY: str
    GEMINI_API_KEY: str
    openai_model: bool 
    gemini_model: bool
    omni_moderation_model: bool

    class Config:
        env_file = ".env"

settings = Settings()
# print(f"OPENAI_API_KEY loaded: {'*' * len(settings.OPENAI_API_KEY)}")